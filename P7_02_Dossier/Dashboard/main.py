import json
import pickle
from urllib.request import urlopen

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import streamlit as st
import streamlit.components.v1 as components

model_load = pickle.load(open('model_shap.md', 'rb'))
shap_values = pickle.load(open('shap_val_f', 'rb'))

features = ['EXT_SOURCE_2',
            'EXT_SOURCE_3',
            'CREDIT_TO_ANNUITY_RATIO',
            'DAYS_BIRTH',
            'PROPORTION_LIFE_EMPLOYED',
            'CREDIT_TO_ANNUITY_RATIO_BY_AGE',
            'DAYS_ID_PUBLISH',
            'DAYS_REGISTRATION',
            'DAYS_EMPLOYED',
            'DAYS_LAST_PHONE_CHANGE',
            'INCOME_TO_ANNUITY_RATIO_BY_AGE',
            'AMT_ANNUITY',
            'INCOME_TO_ANNUITY_RATIO',
            'INCOME_TO_CREDIT_RATIO',
            'AMT_CREDIT',
            'REGION_POPULATION_RELATIVE',
            'AMT_GOODS_PRICE',
            'INCOME_TO_FAMILYSIZE_RATIO',
            'AMT_INCOME_TOTAL',
            'HOUR_APPR_PROCESS_START',
            'AMT_REQ_CREDIT_BUREAU_YEAR',
            'OBS_30_CNT_SOCIAL_CIRCLE',
            'OBS_60_CNT_SOCIAL_CIRCLE',
            'CNT_FAM_MEMBERS',
            'AMT_REQ_CREDIT_BUREAU_QRT']

best_parameters = {'colsample_by_tree': 0.6000000000000001,
                   'learning_rate': 0.026478707430398492,
                   'max_depth': 28.0,
                   'n_estimators': 1000.0,
                   'num_leaves': 4.0,
                   'reg_alpha': 0.8,
                   'reg_lambda': 0.7000000000000001,
                   'solvability_threshold': 0.25,
                   'subsample': 0.8}

st.write('''
# Welcome to the bank loan analysis
###### This application gives the bank's decision for a loan request 
''')

st.sidebar.header("customer's ID")

# Load Dataframe

input_id = st.sidebar.text_input("Please enter the customer's ID", '253286')
input_id = int(input_id)


@st.cache  # mise en cache de la fonction pour exécution unique
def get_data(df):
    dataframe = pd.read_csv(df, index_col=0)
    return dataframe


def get_green_color(text):
    st.markdown(f'<h1 style="color:#23b223;font-size:20px;">{text}</h1>',
                unsafe_allow_html=True)


def get_red_color(url):
    st.markdown(f'<h1 style="color:#ff3333;font-size:20px;">{url}</h1>',
                unsafe_allow_html=True)


def get_API_decision(ID):
    API_url = "https://oc-proejt7.herokuapp.com/predict/" + str(ID)

    # with st.spinner('Chargement du score du client...'):
    json_url = urlopen(API_url)

    API_data = json.loads(json_url.read())
    prediction_result = API_data['credit_bank_proba']
    prediction_decision = API_data['credit_bank_decision']
    return prediction_decision, prediction_result


# def predict(model, id_client, df):
#   data_1 = df.drop('TARGET', axis=1)
#  prediction_result = model.predict_proba(pd.DataFrame(data_1.loc[id_client]).transpose())[:, 1]
#  prediction_bool = np.array((prediction_result > best_parameters['solvability_threshold']) > 0) * 1
# if prediction_bool == 0:
#     prediction_decision = 'Loan approved'
# else:
#    prediction_decision = 'Loan refused'
# return prediction_decision, prediction_result


def get_mean(feature, df, id):
    data_1 = df[df['TARGET'] == 1]
    data_0 = df[df['TARGET'] == 0]
    mean_accepted = data_0[feature].mean()
    mean_denided = data_1[feature].mean()
    feature_client = df[feature][id]

    fig, ax = plt.subplots()
    fig_1 = sns.barplot(x=['mean_accepted', 'mean_refused', 'feature_client'],
                        y=[mean_accepted, mean_denided, feature_client])
    fig_1.set_xticklabels(labels=['mean_accepted', 'mean_refused', 'feature_client'],
                          rotation=45)

    return st.pyplot(fig)


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def get_shap_fig(id_c):
    id_i = {}
    i = 0
    for idx in x_data.index:
        id_i[idx] = i
        i += 1
    figure_shap = shap.force_plot(explainer.expected_value[0], shap_values[1][id_i[id_c]], x_data.iloc[id_i[id_c]],
                                  link="logit")

    return figure_shap


# Appel des fonstion définies:

data = get_data('data_shap.csv')
x_data = data.drop('TARGET', axis=1)

# Si le ID n'est pas dans la liste afficher un message:
if input_id not in data.index.tolist():
    get_red_color(f"The ID {input_id} does not exist! Please enter the customer's ID")
else:

    # Décision de la banque et score:

    bank_decision, probability = get_API_decision(input_id)
    st.subheader(f'Bank decision for customer with ID: {input_id}')

    st.write(f'##### Probability of default: {round(probability * 100, 1)} (%)')
    if bank_decision == 'loan approved':
        get_green_color(f'Bank decision: {bank_decision}')
    else:
        get_red_color(f'Bank decision: {bank_decision}')

    # Affichage des graphes shap pour l'explicabilité du modèle:

    explainer = shap.TreeExplainer(model_load)
    # st.write(explainer.expected_value[0],shap_values[1])

    st.subheader("Features importance")
    st_shap(get_shap_fig(input_id), 200)

    # Comparaison du client avec la moyenne positif et négatif:
    st.subheader("More details for costumer:")
    st.text('We will now compare the value of each feature with mean accepted and mean refused')
    features_options = st.selectbox('Choose the feature you want to compare:', features)
    get_mean(features_options, data, input_id)
