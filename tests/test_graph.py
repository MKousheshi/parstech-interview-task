from consultant_bot.graph import build_graph


def test_say_hello():
    app = build_graph()
    result = app.invoke({"greeting": ""})
    assert result["greeting"] == "Hello, world!"
