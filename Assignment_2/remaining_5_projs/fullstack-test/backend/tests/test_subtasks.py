def test_subtask_lifecycle(client):
    # Create parent task
    task_res = client.post("/api/tasks", json={"title": "Master Task"})
    task_id = task_res.json()["id"]

    # Add subtask
    sub_res = client.post(f"/api/tasks/{task_id}/subtasks", json={"title": "Step 1: Planning"})
    assert sub_res.status_code == 201
    subtask = sub_res.json()
    assert subtask["title"] == "Step 1: Planning"
    assert subtask["is_completed"] is False
    subtask_id = subtask["id"]

    # Verify task reflection
    t = client.get(f"/api/tasks/{task_id}").json()
    assert t["subtask_total"] == 1
    assert t["subtask_completed"] == 0
    assert t["progress_pct"] == 0

    # Update subtask: mark completed
    update_res = client.put(f"/api/subtasks/{subtask_id}", json={"is_completed": True})
    assert update_res.status_code == 200
    assert update_res.json()["is_completed"] is True

    # Check updated progress
    t = client.get(f"/api/tasks/{task_id}").json()
    assert t["subtask_completed"] == 1
    assert t["progress_pct"] == 100

    # Delete subtask
    del_res = client.delete(f"/api/subtasks/{subtask_id}")
    assert del_res.status_code == 200

    t = client.get(f"/api/tasks/{task_id}").json()
    assert t["subtask_total"] == 0
