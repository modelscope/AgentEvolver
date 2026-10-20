"""A stateful MCP server for the OpenWorld connection-lifetime regression test."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("OpenWorld state fixture")
values = []


@server.tool()
def remember(value: str) -> str:
    """Append a value and return this connection's history.

    Args:
        value: Text to remember.

    Returns:
        The accumulated comma-separated values.
    """
    values.append(value)
    return ",".join(values)


if __name__ == "__main__":
    server.run(transport="stdio")
