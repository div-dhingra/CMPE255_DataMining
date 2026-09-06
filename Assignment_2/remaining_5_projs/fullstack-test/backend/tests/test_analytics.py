from datetime import date, timedelta

def test_analytics_empty(client):
    res = client.get("/api/analytics")
    assert res.status_code == 200
    data = res.json()
    assert data["total_tasks"] == 0
    assert data["completed_tasks"] == 0
    assert data["completion_rate"] == 0.0
    assert data["productivity_score"] == 0
    assert len(data["daily_velocity"]) == 7

def test_analytics_with_data(client):
    today = date.today().isoformat()
    # Create tags
    t1 = client.post("/api/tags", json={"name": "Engineering", "color": "#10b981"}).json()["id"]
    t2 = client.post("/api/tags", json={"name": "Marketing", "color": "#f59e0b"}).json()["id"]

    # Create tasks
    task1 = client.post("/api/tasks", json={
        "title": "Build endpoint",
        "priority": "P1",
        "due_date": today,
        "tag_ids": [t1]
    }).json()["id"]

    task2 = client.post("/api/tasks", json={
        "title": "Design banner",
        "priority": "P3",
        "tag_ids": [t2]
    }).json()["id"]

    # Mark task1 as completed
    client.post(f"/api/tasks/{task1}/toggle")

    # Fetch analytics
    res = client.get("/api/analytics")
    assert res.status_code == 200
    data = res.json()

    assert data["total_tasks"] == 2
    assert data["completed_tasks"] == 1
    assert data["active_tasks"] == 1
    assert data["completion_rate"] == 50.0
    assert data["current_streak"] >= 1
    assert data["productivity_score"] > 0

    # Check priority breakdown
    assert data["priority_breakdown"]["P1"] == 1
    assert data["priority_breakdown"]["P3"] == 1

    # Check tag breakdown
    assert len(data["tag_breakdown"]) >= 2
