// Background mock OpenAI-compatible server for integration testing.
import { createServer } from "node:http";
const PORT = Number(process.env.MOCK_PORT || 54330);
const server = createServer((req, res) => {
	if (req.url === "/v1/models") {
		res.setHeader("content-type", "application/json");
		res.end(JSON.stringify({ data: [
			{ id: "mock-llama:7b", context_window: 8192 },
			{ id: "mock-qwen:14b" },
		] }));
	} else {
		res.statusCode = 404; res.end();
	}
});
server.listen(PORT, () => {
	process.stdout.write(`mock-server-up:${PORT}\n`);
});
process.on("SIGTERM", () => server.close(() => process.exit(0)));
