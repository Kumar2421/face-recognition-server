# Kubernetes deployment

Single-node deployment of the face-recognition stack:

| Component      | Kind         | Storage                          | External |
|----------------|--------------|----------------------------------|----------|
| `qdrant`       | StatefulSet  | hostPath PV `data/qdrant_storage`| —        |
| `face-service` | Deployment   | hostPath data dirs + GPU         | NodePort `30001` |
| `face-ui`      | Deployment   | hostPath `ui/` (vite dev)        | NodePort `30173` |

Qdrant is the single stateful container (StatefulSet, 1 replica, stable
storage). `face-service` and `face-ui` are stateless pods whose data lives on
the host via `hostPath` mounts — so the existing `data/`, `images/events/`,
`buffalo_l/models/` content is reused as-is.

## Prerequisites

- A single-node cluster on **this host** (k3s / minikube `--driver=none` / kind):
  hostPath manifests only work where the paths exist.
- **NVIDIA GPU support**: `nvidia-device-plugin` DaemonSet installed, node
  advertising `nvidia.com/gpu`. If your runtime needs it, uncomment
  `runtimeClassName: nvidia` in `30-face-service.yaml`.
- The `face-recognition-server:local` image must be **present on the node**
  (next section).

## 1. Build the app image onto the node

```bash
cd /home/fusion-gpu/fusion-projects/face-recognition-server
docker build -t face-recognition-server:local .

# minikube:  minikube image load face-recognition-server:local
# kind:      kind load docker-image face-recognition-server:local
# k3s:       docker save face-recognition-server:local | sudo k3s ctr images import -
```

`imagePullPolicy: IfNotPresent` — it will not try to pull from a registry.

## 2. (Optional) Migrate existing Qdrant data

Compose stored Qdrant in the docker named volume `*_qdrant_storage`. To keep
your collections, copy it into the hostPath the PV points at:

```bash
src=$(docker volume inspect -f '{{ .Mountpoint }}' face-recognition-server_qdrant_storage)
sudo mkdir -p data/qdrant_storage
sudo cp -a "$src/." data/qdrant_storage/
```

Skip this to start Qdrant empty (collections re-created on demand).

## 3. events.db

`face-service` runs an initContainer `seed-events-db` that copies
`images-old/events/events.db` into the events volume **only if**
`images/events/events.db` does not already exist. Existing live DB is never
overwritten. To force the old snapshot, move the current one aside first:

```bash
sudo mv images/events/events.db images/events/events.db.bak   # then redeploy
```

## 4. Deploy

```bash
kubectl apply -k k8s/
kubectl -n face get pods -w
```

## 5. Access

- API:  `http://<node-ip>:30001`  (health: `/health`)
- UI:   `http://<node-ip>:30173`
- In-cluster: `http://face-service.face:8000`, `http://qdrant.face:6333`

Set the UI's `VITE_API_BASE` (in `40-face-ui.yaml`) to the API URL the
**browser** reaches, e.g. `http://<node-ip>:30001`.

## Notes

- Both app Deployments use `strategy: Recreate` — hostPath/RWO storage and the
  single sqlite writer + single GPU mean exactly one pod at a time. Do **not**
  raise `replicas`.
- `EMBED_THREADS` / `BRANCH_CACHE_TTL` (perf tunables) are in the ConfigMap.
- For a stable external hostname, add an Ingress in front of the `face-service`
  / `face-ui` Services (not included; NodePort is enough for single-node).
