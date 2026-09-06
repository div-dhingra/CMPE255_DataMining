def test_tag_crud(client):
    # Create tag
    res = client.post("/api/tags", json={"name": "Research", "color": "#f43f5e"})
    assert res.status_code == 201
    tag = res.json()
    assert tag["name"] == "Research"
    assert tag["color"] == "#f43f5e"
    tag_id = tag["id"]

    # Duplicate create returns existing
    res_dup = client.post("/api/tags", json={"name": "Research"})
    assert res_dup.status_code == 201
    assert res_dup.json()["id"] == tag_id

    # List tags
    list_res = client.get("/api/tags")
    assert list_res.status_code == 200
    names = [t["name"] for t in list_res.json()]
    assert "Research" in names

    # Delete tag
    del_res = client.delete(f"/api/tags/{tag_id}")
    assert del_res.status_code == 200

    # Verify deleted
    list_res_after = client.get("/api/tags")
    names_after = [t["name"] for t in list_res_after.json()]
    assert "Research" not in names_after
