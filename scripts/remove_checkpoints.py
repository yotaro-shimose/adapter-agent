from oai_utils import gather_with_semaphore
from dotenv import load_dotenv
import asyncio
import tinker


async def main():
    load_dotenv()
    confirm = input("Are you sure you want to delete all checkpoints? (y/n): ")
    if confirm != "y":
        print("Aborting.")
        return
    client = tinker.ServiceClient()
    rest = client.create_rest_client()
    runs = await rest.list_training_runs_async()
    for training_run in runs.training_runs:
        ret = await rest.list_checkpoints_async(training_run.training_run_id)
        await gather_with_semaphore(
            [
                rest.delete_checkpoint_async(
                    training_run.training_run_id, checkpoint.checkpoint_id
                )
                for checkpoint in ret.checkpoints
            ],
            max_concurrent=10,
        )


if __name__ == "__main__":
    asyncio.run(main())
