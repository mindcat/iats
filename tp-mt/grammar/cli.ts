// Batch front-end for Huang & Joo's (2025) toki pona nearley grammar (cubedhuang/ilo-nasin-sin).
// stdin: one JSON string per line; stdout: number of parses per line (0 = ungrammatical).
import { createInterface } from "node:readline";
import { parse } from "../../tp-dataset/corpus/raw/huang-wu/ilo-nasin-sin/src/lib/parser";

const rl = createInterface({ input: process.stdin });
rl.on("line", (line) => {
	let n = 0;
	try {
		n = parse(JSON.parse(line)).length;
	} catch {
		n = 0;
	}
	process.stdout.write(n + "\n");
});
