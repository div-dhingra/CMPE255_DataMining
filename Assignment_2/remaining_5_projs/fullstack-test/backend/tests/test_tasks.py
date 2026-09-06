from datetime import date, timedelta

def test_create_task_basic(client):
    response = client.post("/api/tasks", json={
        "title": "Complete assignment 2",
        "description": "Build high performance fullstack app",
        "priority": "P1",
        "due_date": date.today().isoformat()
    })
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "Complete assignment 2"
    assert data["priority"] == "P1"
    assert data["status"] == "todo"
    assert data["relative_due"] == "Today"
    assert data["is_overdue"] is False
    assert data["id"] is not None

def test_create_task_with_tags_and_subtasks(client):
    # First create a tag
    tag_res = client.post("/api/tags", json={"name": "Backend", "color": "#10b981"})
    assert tag_res.status_code == 201
    tag_id = tag_res.json()["id"]

    response = client.post("/api/tasks", json={
        "title": "Task with dependencies",
        "description": "Includes subtasks and tags",
        "priority": "P2",
        "tag_ids": [tag_id],
        "tag_names": ["DevOps"],
        "subtasks": ["Subtask 1", "Subtask 2"]
    })
    assert response.status_code == 201
    data = response.json()
    assert len(data["tags"]) == 2
    tag_names = [t["name"] for t in data["tags"]]
    assert "Backend" in tag_names
    assert "DevOps" in tag_names
    assert len(data["subtasks"]) == 2
    assert data["subtask_total"] == 2
    assert data["subtask_completed"] == 0
    assert data["progress_pct"] == 0

def test_get_tasks_filtering(client):
    # Create multiple tasks
    client.post("/api/tasks", json={
        "title": "Fix critical security bug",
        "priority": "P1",
        "status": "todo"
    })
    client.post("/api/tasks", json={
        "title": "Update landing page typography",
        "priority": "P3",
        "status": "in_progress"
    })
    client.post("/api/tasks", json={
        "title": "Deploy to staging cluster",
        "priority": "P2",
        "status": "completed"
    })

    # Test status filter: active
    res_active = client.get("/api/tasks?status=active")
    assert res_active.status_code == 200
    assert len(res_active.json()) == 2

    # Test status filter: completed
    res_completed = client.get("/api/tasks?status=completed")
    assert res_completed.status_code == 200
    assert len(res_completed.json()) == 1
    assert res_completed.json()[0]["title"] == "Deploy to staging cluster"

    # Test priority filter
    res_p1 = client.get("/api/tasks?priority=P1")
    assert res_p1.status_code == 200
    assert len(res_p1.json()) == 1
    assert res_p1.json()[0]["title"] == "Fix critical security bug"

    # Test search filter
    res_search = client.get("/api/tasks?search=typography")
    assert res_search.status_code == 200
    assert len(res_search.json()) == 1
    assert res_search.json()[0]["title"] == "Update landing page typography"

def test_update_task(client):
    create_res = client.post("/api/tasks", json={"title": "Draft blog post", "priority": "P4"})
    task_id = create_res.json()["id"]

    update_res = client.put(f"/api/tasks/{task_id}", json={
        "title": "Draft technical blog post on FastAPI",
        "priority": "P2",
        "status": "in_progress"
    })
    assert update_res.status_code == 200
    data = update_res.json()
    assert data["title"] == "Draft technical blog post on FastAPI"
    assert data["priority"] == "P2"
    assert data["status"] == "in_progress"

def test_toggle_task(client):
    create_res = client.post("/api/tasks", json={
        "title": "Finish homework",
        "subtasks": ["Part A", "Part B"]
    })
    task_id = create_res.json()["id"]

    # Toggle to completed
    toggle_res = client.post(f"/api/tasks/{task_id}/toggle")
    assert toggle_res.status_code == 200
    data = toggle_res.json()
    assert data["status"] == "completed"
    assert data["completed_at"] is not None
    # All subtasks should also be marked completed
    assert all(st["is_completed"] for st in data["subtasks"])
    assert data["progress_pct"] == 100

    # Toggle back to todo
    toggle_res2 = client.post(f"/api/tasks/{task_id}/toggle")
    assert toggle_res2.status_code == 200
    assert toggle_res2.json()["status"] == "todo"
    assert toggle_res2.json()["completed_at"] is None

def test_delete_task(client):
    create_res = client.post("/api/tasks", json={"title": "Temporary task to delete"})
    task_id = create_res.json()["id"]

    del_res = client.delete(f"/api/tasks/{task_id}")
    assert del_res.status_code == 200

    get_res = client.get(f"/api/tasks/{task_id}")
    assert get_res.status_code == 404

def test_reorder_tasks(client):
    t1 = client.post("/api/tasks", json={"title": "First"}).json()["id"]
    t2 = client.post("/api/tasks", json={"title": "Second"}).json()["id"]
    t3 = client.post("/api/tasks", json={"title": "Third"}).json()["id"]

    # Reorder t3 to position 0, t1 to position 1, t2 to position 2
    reorder_res = client.post("/api/tasks/reorder", json={
        "orders": [
            {"id": t3, "position": 0},
            {"id": t1, "position": 1},
            {"id": t2, "position": 2}
        ]
    })
    assert reorder_res.status_code == 200

    list_res = client.get("/api/tasks")
    items = list_res.json()
    assert items[0]["id"] == t3
    assert items[1]["id"] == t1
    assert items[2]["id"] == t2
