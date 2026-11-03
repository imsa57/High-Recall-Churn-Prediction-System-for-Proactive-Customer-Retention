from src.data.make_dataset import main as data_main
from src.features.build_features import ChurnPreprocessor
from src.models.train_model import ModelTrainer
from src.models.predict_model import predict_single





def main():
    X_train,X_test,y_train,y_test = data_main()
    preprocessor = ChurnPreprocessor()
    X_transformed = preprocessor.fit_transform(X_train)
    # Example usage
    X_train,X_test,y_train,y_test = data_main()
    print(X_train.shape)
    trainer = ModelTrainer()
    trainer.train_all_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            reference_df=X_train
    )

    result = predict_single(X_test.iloc[1000,:])
    print(result)


if __name__ == "__main__":
    
    main()