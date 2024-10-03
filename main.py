import pandas as pd
from src.model import get_model, predict

def main():
    # Load the Model
    model, tokenizer = get_model()

    # Make predictions on the test set
    index_list, predictions = predict(model, tokenizer, test_file='dataset/test.csv')

    # Prepare output for submission
    output = pd.DataFrame({"index": index_list, "prediction": predictions})
    output.to_csv("test_out.csv", index=False)

    print("Predictions saved to test_out.csv")

if __name__ == "__main__":
    main()
