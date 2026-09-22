from consultant_bot.graph import build_graph


def main() -> None:
    app = build_graph()
    result = app.invoke({"greeting": ""})
    print(result["greeting"])
