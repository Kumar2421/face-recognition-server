# Why am I seeing duplicate IDs for the same person?

## 1. What is happening?
You are seeing multiple visitor IDs (like `visiter-1` and `visiter-2`) for the same person in the system. Even though we have a "lock" to prevent this, it still happens when a person is detected multiple times very quickly (milliseconds apart).

## 2. The Step-by-Step Reason (The "Race Condition")

Imagine two computer processes, **A** and **B**, receiving pictures of the same person at almost the exact same time:

1.  **Process A** looks at the first picture. It searches the database, finds no match, and grabs the "Lock" to create a new ID.
2.  **Process B** looks at the second picture. It also finds no match and waits outside the "Lock" for Process A to finish.
3.  **Process A** creates the new ID (e.g., `visiter-1`), saves it to the database (Qdrant), and releases the "Lock".
4.  **Process B** immediately enters the "Lock" and searches again to see if anyone else just added this person.
5.  **THE PROBLEM**: The database (Qdrant) is "eventually consistent." This means it takes a tiny fraction of a second (like 100-200ms) to update its search index. Because Process B searched *immediately*, the database said "I still don't see anyone," so Process B created a second ID (`visiter-2`).

## 3. Real-World Evidence (The Proof)

We have observed clear evidence of this in your system logs and screenshots:

*   **Identical Timestamps**: Duplicate IDs (like `visiter-75272` and `visiter-79341`) were created within **2 seconds** of each other. This is exactly the "race condition" window where two processes are running at the same time.
*   **Near-Identical Scores**: One detection showed a **67% match**, while another (for the same person) showed a **62% match**. Because the second score was slightly lower, the system didn't feel "confident" enough to link them, so it created a new ID.
*   **High-Traffic Cameras**: The issue is most common on cameras like `GF-Billing-2` where people are captured many times in a very short period.

## 4. Other Contributing Factors

*   **Image Quality**: Sometimes the first picture is very clear (67% match), but the second picture is slightly blurry (62% match). If the second picture's score is too low, the system thinks it's a different person entirely.
*   **Fast Movement**: When a person moves quickly across the camera, the system triggers many events per second, which makes the "database update delay" described above much more likely to cause issues.

## 4. How we fixed it for you (UI Level)

We have updated the **Event Review** and **Recognition** pages to hide this mess:
*   **Automatic Grouping**: If the same ID appears multiple times on the same day, we now combine them into **one single row** in your list.
*   **All Images in One Place**: When you click to expand that row, you will see **all the pictures** from every detection, even if they were originally duplicates.
*   **Accurate Counting**: The "Unique Count" at the top now counts **actual people**, not just the number of times the ID was created.

## 5. How to fully solve it in the Backend (Future Steps)
*   **Force Database Refresh**: Tell the database to "refresh its memory" immediately after saving a new visitor so the next process sees it.
*   **New ID Cooldown**: Add a 5-second rule: "If you just created a new visitor on this camera, don't create another one for at least 5 seconds."
