import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier


def main():
    st.title("Multi class classification")
    st.subheader("Predicting the drop out rate of students")
    st.markdown("This dataset contains data from a higher education institution on various variables related to undergraduate students, including demographics, social-economic factors, and academic performance, to investigate the impact of these factors on student dropout and academic success")
    st.sidebar.title("Multi class classification")
    st.sidebar.markdown("Predicting the drop out rate of students")

    # fetch dataset
    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv("dataset.csv")
        labelencoder = LabelEncoder()
        for col in data.columns:
            data[col] = labelencoder.fit_transform(data[col])
        return data

    df = load_data()
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Student Data")
        st.write(df)

    target_mapping = {0: "Enrolled", 2: "Graduate", 1: "Dropout"}
    df['Target_mapped'] = df['Target'].map(target_mapping)

    def visualizations(charts):
        if "Correlation Matrix" in charts:
            df1= df.drop(["Target_mapped"], axis=1)
            corr_matrix = df1.corr()
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(corr_matrix, cmap="coolwarm", ax=ax, annot=True, fmt=".2f")
            ax.set_title("Correlation Heatmap between variables")
            st.pyplot(fig)


        if "Gender Distribution" in charts:
            st.markdown("The gender of the student")
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.countplot(x="Gender", data=df,hue="Target_mapped")
            plt.xticks(ticks=[1,0],labels=["Male","Female"])
            ax.set_title("Gender Distribution")
            st.pyplot(fig)

        if "Marital status" in charts:
            st.markdown("The marital status of the student")
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.countplot(x="Marital status", data=df,hue="Target_mapped")
            plt.xticks(ticks=[0,1,2,3,4,5],labels=['Single','Married','Widower','Divorced','Facto union','Legally separated'])
            st.pyplot(fig)
        if "Course" in charts:
            st.markdown("The course taken by the student")
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.countplot(x="Course", data=df, hue="Target_mapped")
            plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], rotation=45,labels=['Biofuel Production Technologies','Animation and Multimedia Design','Social Service','Agronomy','Communication Design','Veterinary Nursing','Informatics Engineering','Equiniculture','Management','Social Service','Tourism','Nursing','Oral Hygiene','Advertising and Marketing Management','Journalism and Communication','Basic Education','Management'])
            st.pyplot(fig)

        if "Scholarship holder" in charts:
            st.markdown("Whether the student is a scholarship holder")
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.countplot(x="Scholarship holder", data=df, hue="Target_mapped")
            plt.xticks(ticks=[0,1],labels=["no","yes"])
            st.pyplot(fig)

        if "Debtors" in charts:
            st.markdown("Whether the student is a debtor")
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.countplot(x="Debtor", data=df, hue="Target_mapped")
            plt.xticks(ticks=[0,1],labels=["no","yes"])
            st.pyplot(fig)


    st.sidebar.subheader("Choose visualizations")
    visuals = st.sidebar.radio("Select Visualizations to Display", ["Correlation Matrix","Gender Distribution","Marital status","Course","Scholarship holder","Debtors"])
    if st.sidebar.button("Show Visuals", key='visuals_button'):
        if visuals:
            visualizations(visuals)
        else:
            st.warning("Please select at least one visualization.")

    class_names = ['Enrolled','Graduate', 'Dropout']
    @st.cache_data(persist=True)
    def split(df):
        x = df.drop(["Target","Target_mapped"], axis=1)
        y = df['Target']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot()
            st.pyplot(plt)
            plt.clf()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(x_test)
                for i in range(len(class_names)):
                    fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
                    plt.plot(fpr, tpr, label=f"ROC curve for {class_names[i]}")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend(loc="best")
                st.pyplot(plt)
                plt.clf()  # Clear the current figure for the next plot
            else:
                st.warning("ROC Curve is only available for models that support probability estimates.")

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(x_test)
                for i in range(len(class_names)):
                    precision, recall, _ = precision_recall_curve(y_test == i, y_pred_proba[:, i])
                    plt.plot(recall, precision, label=f"PR curve for {class_names[i]}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall Curve")
                plt.legend(loc="best")
                st.pyplot(plt)
                plt.clf()
            else:
                st.warning("Precision-Recall Curve is only available for models that support probability estimates.")


    x_train, x_test, y_train, y_test = split(df)
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.radio("Classifier",
                                      ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        st.subheader("Hypermeters details")
        st.markdown("Regularization parameter(C): regularizes the training loss of misclassified data")
        st.markdown("Kernels: mathematical functions that transform the data into a higher-dimensional space (rbf - It is used to perform transformation when there is no prior knowledge about data, linear - used when the data is Linearly separable, that is, it can be separated using a single Line.  ")
        st.markdown("Gamma: Kernel coefficient for rbf,poly and sigmoid functions. (auto - 1/n_features, scale - 1 / (n_features * X.var())")
        # choose parameters
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
            model = OneVsRestClassifier(model)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 3))
            st.write("Precision: ", round(precision_score(y_test, y_pred, average="weighted"), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, average="weighted"), 2))
            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        st.subheader("Hypermeters details")
        st.markdown("Regularization parameter(C): regularizes the training loss of misclassified data")
        st.markdown("n_estimators: The number of trees in the forest")
        st.markdown("max_depth: The number of trees in the forest")
        st.markdown("bootstrap: Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree")
        n_estimators = st.sidebar.number_input("n_estimators", 100, 5000, step=10,
                                               key='n_estimators')
        max_depth = st.sidebar.number_input("max_depth", 1, 20, step=1, key='max_depth')

        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        bootstrap = True if bootstrap == 'True' else False
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap,
                                           n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 3))
            st.write("Precision: ", round(precision_score(y_test, y_pred,average="weighted"),2))
            st.write("Recall: ", round(recall_score(y_test, y_pred,average="weighted"),2))
            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        st.subheader("Hypermeters details")
        st.markdown("Regularization parameter(C): regularizes the training loss of misclassified data")
        st.markdown("Maximum number of iterations: Maximum number of iterations taken for the solvers to converge")
        st.markdown("class_weight: Weights associated with classes in the form (none - all classes are supposed to have weight one, balanced -the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)) ")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        class_weight = st.sidebar.radio("class_weight", ("balanced", None), key='class_weight')
        metrics = st.sidebar.multiselect("What metrics to plot?",
                                         ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter,class_weight=class_weight)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 3))
            st.write("Precision: ", round(precision_score(y_test, y_pred, average="weighted"), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, average="weighted"), 2))
            plot_metrics(metrics)

if __name__ == '__main__':
    main()
