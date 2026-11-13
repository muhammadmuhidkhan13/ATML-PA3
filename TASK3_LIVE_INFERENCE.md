# Task 3 live record protocol

The frozen assets contain the first 50 MATH evaluation IDs, the exact original questions and gold references, the model-file lock, and the pinned scorer and generation protocol. The model runs unquantized FP16 on CUDA at batch one with SDPA and its native chat template. The live strategy schedule is one 3,072-token call, three 1,024-token parallel calls, and one 2,048-token revision of the first parallel response. Seeds derive from the global seed 6304, question ID and call name.

The runner records each call's rendered input, raw output, generated token IDs, extraction, inference time, peak allocated and reserved memory, cap and EOS flags and hardware/software provenance. Grading and completion validation reconstruct the strategy outcomes from hashed call records and independent per-question grades. Parallel self-consistency groups answers using the frozen semantic scorer. Both parallel and refinement are charged for the shared initial response in their strategy totals. The physical execution totals count the shared call once.

Each record is written atomically. On resume, compatible call and grade records are checked and reused. If an interrupted process left a run lock, verify the original process has exited before clearing its stale lock. A completed official run includes 250 calls, 50 linked grades, 150 strategy results and the verified preflight proof. CPU export independently checks these records and generates the tables, qualitative examples and seven figures.

See [deployment commands](TASK3_DEPLOYMENT.md).
