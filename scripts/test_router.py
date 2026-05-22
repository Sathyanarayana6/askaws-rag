"""Unit test for the Router Agent."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.state import empty_state
from src.agents.router import router_node

TEST_QUESTIONS = [
    # Should retrieve
    ("How do I create an S3 bucket?", "retrieve"),
    ("What is the difference between IAM users and roles?", "retrieve"),
    ("How do I trigger a Lambda function from S3?", "retrieve"),
    ("What are EC2 spot instances?", "retrieve"),
    ("How does SageMaker Feature Store work?", "retrieve"),

    # Should refuse — off-topic
    ("What's the weather in Allen Texas?", "refuse"),
    ("Write me a poem about pizza", "refuse"),
    ("hi", "refuse"),

    # Should refuse — out-of-scope AWS services
    ("How do I set up DynamoDB?", "refuse"),
    ("How do I use CloudFront?", "refuse"),
]


def main():
    print("=" * 70)
    print("Router Agent Smoke Test")
    print("=" * 70)

    correct = 0
    total = len(TEST_QUESTIONS)

    for question, expected in TEST_QUESTIONS:
        state = empty_state(question)
        update = router_node(state)
        actual = update["route_decision"]
        ok = actual == expected
        correct += int(ok)

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] expected={expected:8s} got={actual:8s} | {question}")
        if not ok:
            print(f"        Reason: {update.get('routing_reason')}")

    print("\n" + "=" * 70)
    print(f"Results: {correct}/{total} correct ({100 * correct / total:.0f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
