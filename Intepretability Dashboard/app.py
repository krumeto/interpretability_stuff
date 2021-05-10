import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance, plot_partial_dependence
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

st.title('Partial Dependence Plots Dashboard')
"""
### Step 1: Import Data
"""
data_input = st.radio('Slide to select', options=['Use the example dataset','Upload yours'])

if data_input == 'Upload yours':
    data = st.file_uploader('Import your own data, but please let it be in a csv format', type='csv')


data = pd.read_csv('data/penguins.csv')
st.write(data.head())
st.write('Percentage of NA values')
st.write(data.isna().mean())

"""
### Step 2: Select a target variable
"""
st.write('Please select a target variable')

target = st.selectbox('Target variable', data.columns, help = 'Please select a target variable')

#data = data.dropna(subset=[target])
data = data.dropna()

X = data.drop(target, axis=1)
print(X.head())
print(X.dtypes)

n_categorical_features = ((X.dtypes == 'category') | (X.dtypes == 'object')).sum()
print(n_categorical_features)
n_numerical_features = (X.dtypes == 'float').sum()
le = LabelEncoder()
y = le.fit_transform(data[target])
# The ordinal encoder will first output the categorical features, and then the
# continuous (passed-through) features
categorical_mask = ([True] * n_categorical_features +
                    [False] * n_numerical_features)

print(categorical_mask)
ordinal_encoder = make_column_transformer(
    (OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan),
     make_column_selector(dtype_include=['category', 'object'])),
    remainder='passthrough')

clf = make_pipeline(
    
    ordinal_encoder,
    SimpleImputer(strategy='most_frequent'),
    HistGradientBoostingClassifier(random_state=42,
                                  categorical_features=categorical_mask)
)
scores = cross_val_score(clf, X, y, cv=5)

st.header('Cross-validation Results')
st.write("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

clf.fit(X,y)

"""
### Step 3: Create partial dependency plots.

Partial dependence plots (PDP) and individual conditional expectation (ICE) plots can be used to visualize and analyze interaction between the target response 1 and a set of input features of interest.

Intuitively, we can interpret the partial dependence as the expected target response as a function of the input features of interest.
"""

st.header('Partial dependency Plots. Please select variables from the sidebar')

if len(le.classes_) > 2:
    pdp_target = st.selectbox('Multiclass classification, please select a target', np.unique(y))
    st.write(f'Selected: {le.inverse_transform([int(pdp_target)])}')



features = st.sidebar.multiselect(
    "Select Features for PDPs",
    X.columns
)

multi_features = st.sidebar.multiselect(
    "If you'd like to combine two , Select Multi-Features",
    X.columns
)


for f in features:
    st.write(f)
    ice_plots = st.selectbox(
    f'Plot type for feature {f}:', 
    ["average", "individual", "both"])
    fig, ax = plt.subplots()
    
    if len(le.classes_) > 2:
        plot_partial_dependence(clf, X, [f], target = [int(pdp_target)], kind = ice_plots,ax=ax) 
    else:
        plot_partial_dependence(clf, X, [f], kind = ice_plots, ax=ax)
    st.pyplot(fig)

if multi_features:
    ice_plots = st.selectbox(
        f'Plot type for {multi_features}:', 
    ["average", "individual", "both"])
    fig, ax = plt.subplots()    
    if len(le.classes_) > 2:
        plot_partial_dependence(clf, X, [multi_features], target=pdp_target, kind = ice_plots, ax=ax) 
    else:
        plot_partial_dependence(clf, X, [multi_features], kind = ice_plots, ax=ax) 
    st.pyplot(fig)