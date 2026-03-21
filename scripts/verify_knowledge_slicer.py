import asyncio
import logging

from oai_utils.agent import AgentsSDKModel

from adapter_agent.hierarchical.agent.knowledge_slicer import KnowledgeSlicer
from adapter_agent.model_helper import get_gemini

logging.basicConfig(level=logging.INFO)

SAMPLE_KNOWLEDGE = """
The `numrs2` library provides a high-performance `Array` structure for numerical computations. 
To create a 2D array, use `Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2,2])`.
Matrix multiplication is performed using the `matmul` method: `a.matmul(&b)`.

```rust
use numrs2::array::Array;

fn main() {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2,2]);
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2,2]);
    let c = a.matmul(&b);
    println!("{:?}", c);
}
```
"""


async def test_slicer():
    model = get_gemini()
    slicer = KnowledgeSlicer(model=model)

    print("Generating QRAs...")
    qras = await slicer.slice(SAMPLE_KNOWLEDGE)

    print(f"Generated {len(qras)} QRAs:")
    for i, qra in enumerate(qras):
        print(f"\n--- QRA {i + 1} ---")
        print(f"QUESTION:\n{qra.question}")
        print(f"\nREASONING:\n{qra.reasoning}")
        print(f"\nANSWER:\n{qra.answer}")


if __name__ == "__main__":
    asyncio.run(test_slicer())
