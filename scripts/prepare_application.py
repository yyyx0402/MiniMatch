"""Prepare the locked current model for the local product application."""
from productsearch.release import prepare_release

if __name__ == "__main__":
    manifest = prepare_release()
    print(f"Application ready: {manifest['model_name']} / {manifest['default_search']}")
