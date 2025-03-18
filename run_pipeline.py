from pipelines.training_pipeline import train_pipeline
from src.steps.clean_data import clean_data
from src.steps.evaluation import evaluation
from src.steps.ingest_data import ingest_data
from src.steps.model_train import train_model


if __name__ == "__main__":
    training = train_pipeline(
        ingest_data(),
        clean_data(),
        train_model(),
        evaluation(),
    )

    training.run()

  