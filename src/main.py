from src.data_loader import load_train_data, load_test_data

train_dir = "dataset/train"
test_dir = "dataset/test"

def load_data(train_dir,test_dir):
    
    source1_train, source2_train, source3_train, ground_truth = (
        load_train_data(train_dir)
    )

    source1_test, source2_test, source3_test = (
        load_test_data(test_dir)
    )

    print("Training data:")
    print("Source 1:", source1_train.shape)
    print("Source 2:", source2_train.shape)
    print("Source 3:", source3_train.shape)
    print("Ground truth:", ground_truth.shape)

    print("\nTest data:")
    print("Source 1:", source1_test.shape)
    print("Source 2:", source2_test.shape)
    print("Source 3:", source3_test.shape)

    print(f"Sources_Columns: {source1_train.columns}")
    print(f"Ground_truth_Columns: {ground_truth.columns}")


    return (
        source1_train,
        source2_train,
        source3_train,
        ground_truth,
        source1_test,
        source2_test,
        source3_test
    )

if __name__ == "__main__":
    load_data(train_dir,test_dir)