# Steam-Game-Recommender-kNN
Content-based recommendation system for Steam games using k-Nearest Neighbors (87k dataset) with CustomTkinter GUI.

# ML Game Recommender

A content-based recommendation system that suggests 10 similar video games based on user input. The model analyzes game characteristics using the k-Nearest Neighbors (k-NN) algorithm and features a modern GUI built with CustomTkinter.

## Tech Stack
*   **Language:** Python
*   **Machine Learning:** Scikit-learn (k-NN, Clustering, Scaling), Pandas, NumPy
*   **GUI:** CustomTkinter

## Dataset
The model was trained on the **Steam 2024 Dataset** from Kaggle, containing over 87,000 games.
*(Note: Due to size limitations, the dataset and the pre-trained `.pkl` model are not included in this repository. You can download the dataset from Kaggle and run the training script).*

## Project Pipeline
1.  **Data Cleaning:** Handling missing values and converting all parameters to numeric formats.
2.  **Feature Engineering:** Removing highly correlated features and reducing dimensionality.
3.  **Clustering & Scaling:** Grouping data and normalizing feature scales for better distance calculation.
4.  **Model Building:** Implementing the k-Nearest Neighbors algorithm due to its efficiency in finding object similarity.
5.  **GUI Development:** Creating an interactive interface for users to input a game and receive recommendations.
6.  **Evaluation:** Testing recommendation accuracy based on game characteristics.

## How to Run
1. Clone the repository:
``bash
git clone https://github.com/markblayd/ML-Game-Recommender.git
``
2. Install the required libraries:
``bash
pip install -r requirements.txt
``
3. Run the application:
``bash
python impo.py
``

## Challenges & Learnings
*   **High-Dimensional Data:** Dealing with a large dataset (87k entries) required careful feature selection and dimensionality reduction to make k-NN work efficiently.
*   **Feature Correlation:** Identifying and removing abnormally correlated features to improve the model's objective similarity metrics.
*   **UI/UX Integration:** Successfully bridging a backend ML model with a user-friendly GUI (CustomTkinter) to make the system accessible to end-users.

## Future Improvements
*   Incorporate user ratings/ratings into the similarity algorithm.
*   Improve algorithm precision and processing speed.
