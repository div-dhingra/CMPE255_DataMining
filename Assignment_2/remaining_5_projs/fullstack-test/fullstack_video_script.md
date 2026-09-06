# TodoPro - End-to-End Video Demo Script

**Target Duration:** ~3 Minutes
**Objective:** Visually bridge the gap between core backend/frontend code logic and the interactive UI/UX features it produces.

---

## 🎬 Introduction (0:00 - 0:20)
- **Visuals:** Start on the main TodoPro dashboard, showing a populated list of tasks, tags, and the analytics header.
- **Narration:** "Welcome to TodoPro, a modern full-stack dynamic task management platform. Today, we're going to dive under the hood to see how three core features in the code map directly to what you see here on the UI."

---

## 1️⃣ Feature 1: Dynamic Task Reordering (0:20 - 1:00)

**The Code Context:**
- **Visuals:** Open your IDE to `frontend/js/app.js` (around line 551). Point out the `handleCardDrop` method, explaining that this vanilla JavaScript calculates the new index array.
- **Visuals:** Next, jump to `backend/app/crud.py` (around line 285). Highlight the `reorder_tasks` function, which accepts the array of reordered IDs and persists their new `position` integers to the SQLite database.

**The UI Connection:**
- **Visuals:** Switch back to the browser.
- **Action:** Grab the drag handle (`⋮⋮`) on one of the task cards. Drag it up and drop it in a new position.
- **Narration:** "When we grab this drag handle and move a task, `handleCardDrop` in the frontend triggers. It sends the new positional data to the backend's `reorder_tasks` function, which instantly updates the database so the new order persists perfectly across page reloads."

---

## 2️⃣ Feature 2: Interactive Subtask Progress Calculation (1:00 - 1:40)

**The Code Context:**
- **Visuals:** In the IDE, open `backend/app/crud.py` (around line 62).
- **Narration:** "Look at the `enrich_task_data` function. Notice how it dynamically tallies up completed subtasks vs total subtasks to mathematically calculate a live `progress_pct` (progress percentage) before sending the data to the client."

**The UI Connection:**
- **Visuals:** Switch back to the browser and look at a task card that contains subtasks.
- **Action:** Click to complete an inline subtask checkbox. Watch the animated progress bar update and the text change (e.g., from `1 of 3 done` to `2 of 3 done`). 
- **Narration:** "On the UI, this backend calculation drives the real-time progress meter inside the task card. Every time a subtask is toggled, the database recalibrates the percentage, immediately filling this visual progress bar."

---

## 3️⃣ Feature 3: Productivity Analytics Dashboard Engine (1:40 - 2:40)

**The Code Context:**
- **Visuals:** Open `backend/app/crud.py` (around line 391) to the `get_analytics` function.
- **Narration:** "Finally, let's look at the analytics engine. This large function analyzes your historical data. It counts your current streak of continuous days completing tasks, calculates your 7-day velocity, and derives a holistic Productivity Score out of 100 based on your completion rate and overdue penalties."

**The UI Connection:**
- **Visuals:** Switch to the browser and highlight the top analytics dashboard.
- **Action:** Point out the Productivity Score gauge, the Streak counter (`🔥`), and hover over the 7-Day Velocity interactive bar chart to show tooltips. Click a task checkbox to complete it (triggering the confetti and chime).
- **Narration:** "All that mathematical heavy lifting in the backend maps directly to this rich dashboard at the top of the app. Your completion streak, velocity chart, and overall score dynamically respond to your work. Completing a task instantly updates these metrics—paired with a rewarding burst of confetti and audio chimes."

---

## 🏁 Conclusion (2:40 - 3:00)
- **Visuals:** Zoom out to show the whole dashboard working cohesively in Dark Mode.
- **Narration:** "By combining a clean FastAPI backend with a highly interactive vanilla JavaScript frontend, TodoPro delivers an incredibly fast and feature-rich user experience. Thanks for watching."
