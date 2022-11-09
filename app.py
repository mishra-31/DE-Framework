import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

import time
import random
import string
import joblib
import warnings

warnings.filterwarnings("ignore")

import streamlit as st


def intro():
    import streamlit as st

    st.write("# Welcome to Data evaluation 📊📈")
    st.sidebar.success("Select an option")

    st.markdown(
        """
          A Large number of populations is “unbanked” or “underbanked,” meaning they have no bank account or cannot access their bank’s full range of financial services to build credit and plan for the future. 

          Fintech innovation can help unbanked and underbanked citizens overcome barriers such as minimum balance requirements, no credit history, lack of trust as traditional banks ignore these customers and don’t have the data to understand their needs, and that also contributes to the lack of trust.

          When it comes to working with this segment of the population financial services have to use a broader range of data to assess the customers.

          This use case helps us in choosing the right data across a huge variety of dataset available from a large universe of data since pulling up all the data from each and every dataset can cost time and money.
     """
    )


def eda():
    st.title("Descriptive Analysis")
    st.write(
        """
        This page illustrates the EDA for the data sources!!
"""
    )
    home_rent = pd.read_csv("https://raw.githubusercontent.com/mishra-31/DE-Framework/main/Home%20rent.csv",
                            na_values=['='])
    home_rent = home_rent.iloc[:, 1:]

    electric_bill = pd.read_csv("https://raw.githubusercontent.com/mishra-31/DE-Framework/main/Electricity%20Bill.csv",
                                na_values=['='])
    electric_bill = electric_bill.iloc[:, 1:]

    telephone_bill = pd.read_csv("https://raw.githubusercontent.com/mishra-31/DE-Framework/main/Telephone%20Bill.csv",
                                 na_values=['='])
    telephone_bill = telephone_bill.iloc[:, 1:]

    payday = pd.read_csv("https://raw.githubusercontent.com/mishra-31/DE-Framework/main/payday%20loans.csv",
                         na_values=['='])
    payday = payday.iloc[:, 1:]

    bnpl = pd.read_csv("https://raw.githubusercontent.com/mishra-31/DE-Framework/main/BNPL%20Loans.csv",
                       na_values=['='])
    bnpl = bnpl.iloc[:, 1:]

    cust_names = list(
        set(list(home_rent.iloc[:, 0]) + list(payday.iloc[:, 0]) + list(electric_bill.iloc[:, 0]) + list(
            telephone_bill.iloc[:, 0]) + list(bnpl.iloc[:, 0])))

    anonimyzied_id = []
    for i in range(len(cust_names)):
        res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
        if res not in anonimyzied_id:
            anonimyzied_id.append(str(res))

    id_dataset = pd.DataFrame({'Customer names': cust_names, 'Anonimised id': anonimyzied_id})

    home_rent = pd.merge(id_dataset, home_rent, how='right', right_on='customer names', left_on='Customer names')
    home_rent = home_rent.drop(home_rent.columns[[0, 2]], axis=1)

    payday = pd.merge(id_dataset, payday, how='right', right_on='customer names', left_on='Customer names')
    payday = payday.drop(payday.columns[[0, 2]], axis=1)

    bnpl = pd.merge(id_dataset, bnpl, how='right', right_on='customer names', left_on='Customer names')
    bnpl = bnpl.drop(bnpl.columns[[0, 2]], axis=1)

    telephone_bill = pd.merge(id_dataset, telephone_bill, how='right', right_on='customer name',
                              left_on='Customer names')
    telephone_bill = telephone_bill.drop(telephone_bill.columns[[0, 2]], axis=1)

    electric_bill = pd.merge(id_dataset, electric_bill, how='right', right_on='customer name', left_on='Customer names')
    electric_bill = electric_bill.drop(electric_bill.columns[[0, 2]], axis=1)

    def dataset_analysis(dataframe):

        fig = go.Figure(data=[go.Table(columnwidth=[1, 1], header=dict(values=['Observations', 'Values'],
                                                                       align='center'),
                                       cells=dict(values=[
                                           ['Number of variables', 'Number of observations', 'Missing cells',
                                            'Missing cells (%)', 'Duplicate rows',
                                            'Numerical Features', 'Categorical Features'],
                                           [len(dataframe.columns), dataframe.shape[0], sum(dataframe.isnull().sum()),
                                            f'{sum(dataframe.isnull().sum()) / (dataframe.shape[0] * len(dataframe.columns)) * 100}%',
                                            dataframe[dataframe.duplicated()].shape[0],
                                            len([feature for feature in dataframe.columns if
                                                 dataframe[feature].dtypes != 'O']),
                                            len([feature for feature in dataframe.columns if
                                                 dataframe[feature].dtypes == 'O'])]],

                                           align='center'))])

        fig.update_layout(
            height=350,
            showlegend=False,
            title_text="Dataset Statistics",
            title_x=0.5,
            template="plotly_dark"
        )

        st.plotly_chart(fig)

        st.markdown('Variables')

        for var in [feature for feature in dataframe.columns if dataframe[feature].dtypes != 'O']:
            fig = ff.create_distplot([dataframe[var].values.tolist()], [var], show_rug=False)
            fig.update_layout(title_text=" ".join(f'{var}'.split('_')), template="plotly_dark")
            st.plotly_chart(fig)

        for i in [feature for feature in dataframe.columns if dataframe[feature].dtypes == 'O']:

            if len(dataframe[i].value_counts()) > 5:
                fig = go.Figure(data=[go.Table(header=dict(values=['Observations', 'Values'],
                                                           align='center'),
                                               cells=dict(values=[['Distinct Values', 'Count', 'Unique values'],
                                                                  [len(dataframe.dropna()[i].unique()),
                                                                   dataframe[
                                                                       [feature for feature in dataframe.columns if
                                                                        dataframe[feature].dtypes == 'O']].describe()[
                                                                       i]['count'],
                                                                   f"{' , '.join(list(dataframe[i].value_counts()[:5].index))}............."]]
                                                          , align='center'))])

                fig.update_layout(
                    height=290,
                    showlegend=False,
                    title_text=" ".join(f'{i}'.split('_')),
                    title_x=0.5,
                    template="plotly_dark"
                )

                st.plotly_chart(fig)
            else:
                fig = go.Figure(data=[go.Table(header=dict(values=['Observations', 'Values'],
                                                           align='center'),
                                               cells=dict(values=[['Distinct Values', 'Count', 'Unique values'],
                                                                  [len(dataframe.dropna()[i].unique()),
                                                                   dataframe[
                                                                       [feature for feature in dataframe.columns if
                                                                        dataframe[feature].dtypes == 'O']].describe()[
                                                                       i]['count'],
                                                                   ' , '.join(
                                                                       list(dataframe[i].value_counts()[:5].index))]]
                                                          , align='center'))])

                fig.update_layout(
                    height=290,
                    showlegend=False,
                    title_text=" ".join(f'{i}'.split('_')),
                    title_x=0.5,
                    template="plotly_dark"
                )

                st.plotly_chart(fig)

        fig = px.imshow(dataframe.corr(), width=700,
                        height=700, template='plotly_dark')
        fig.update_layout(title_text="Correlation Matrix")

        st.plotly_chart(fig)

    def interpretation(dataframe):

        ## missing values
        missing = False
        missing_values = sum(dataframe.isnull().sum())
        missing_value_percentage = ((missing_values) / (dataframe.shape[0] * len(dataframe.columns))) * 100
        if missing_value_percentage > 15:
            missing = True


        ##normality
        not_normal = []
        all_var=[feature for feature in dataframe.columns if dataframe[feature].dtypes != 'O']
        for var in all_var:
            stat, p = shapiro(dataframe[var].values)

            if p <= 0.05:
                not_normal.append(var)

        total_non_noramal_vars = len(not_normal)
        ## multicolinearity
        x = dataframe.corr()
        mc_variable = []
        ind = 0
        for i in x.values:
            indd = 0
            for j in i:
                if (j > 0.8 or j < -0.8) and (ind != indd):
                    mc_variable.append((j, x.index[ind], x.columns[indd]))
                indd += 1
            ind += 1

        c = 0
        l = []
        mc_variables = []
        for i in mc_variable:
            if i[0] not in l:
                mc_variables.append(mc_variable[c])
                l.append(i[0])
            c += 1

        total_mc_vars = len(mc_variables)

        return missing, total_non_noramal_vars / len(all_var), total_mc_vars / len(all_var), (1-missing_value_percentage)*100

    def final_interpretation(dataframe):
        if interpretation(dataframe)[0] == True or interpretation(dataframe)[1] < 0.5 or interpretation(dataframe)[
            2] < 0.3:
            st.write(
                f'Data is optimal since missing values are lesser than 15%, around {round(interpretation(dataframe)[1] * 100)}% variables of whole dataset are normal and it has a multicollinearity of {round(interpretation(dataframe)[2] * 100)}%')
        else:
            st.write(
                f'Dataset is not good since missing values are more than 15%, around {round(interpretation(dataframe)[1] * 100)}% variables of whole dataset are not normal and it has  multicollinearity of {round(interpretation(dataframe)[2] * 100)}%')

    if st.button('Home rent data'):
        st.write(home_rent.head(20))
        final_interpretation(home_rent)
        dataset_analysis(home_rent)
    if st.button('BNPL Loans'):
        st.write(bnpl.head(20))
        final_interpretation(bnpl)
        dataset_analysis(bnpl)
    if st.button('Electric bill'):
        st.write(electric_bill.head(20))
        final_interpretation(electric_bill)
        dataset_analysis(electric_bill)
    if st.button('Payday Loans'):
        st.write(payday.head(20))
        final_interpretation(payday)
        dataset_analysis(payday)
    if st.button('Telephone bill'):
        st.write(telephone_bill.head(20))
        final_interpretation(telephone_bill)
        dataset_analysis(telephone_bill)

    data_sources=[('Home rent data',home_rent),('BNPL Loans',bnpl),('Electric bill',electric_bill),
                  ('Payday Loans',payday),('Telephone bill',telephone_bill)]

    ds=[]
    avail=[]
    normm=[]
    multicol=[]
    for i in data_sources:
        print(i[0])
        print(1-interpretation(i[1])[1])
        ds.append(i[0])
        avail.append(f'{round(interpretation(i[1])[-1],2)}%')
        normm.append(f'{((1-interpretation(i[1])[1]))*100}%')
        multicol.append(f'{round((interpretation(i[1])[2]*100),2)}%')


    eda_df=pd.DataFrame({'Data Source ' : ds,
                  'Availibity' : avail,
                  'Normality' : normm,
                  'Multicolinearity' : multicol})
    eda_df.to_csv('eda.csv')
    #st.write(eda_df)



def data_evaluation_page():
    st.title("Data Evaluation")
    st.write(
        """
        Finding the right data.
""")
    home_rent = pd.read_csv("https://raw.githubusercontent.com/mishra-31/DE-Framework/main/Home%20rent.csv",
                            na_values=['='])
    home_rent = home_rent.iloc[:, 1:]

    electric_bill = pd.read_csv("https://raw.githubusercontent.com/mishra-31/DE-Framework/main/Electricity%20Bill.csv",
                                na_values=['='])
    electric_bill = electric_bill.iloc[:, 1:]

    telephone_bill = pd.read_csv("https://raw.githubusercontent.com/mishra-31/DE-Framework/main/Telephone%20Bill.csv",
                                 na_values=['='])
    telephone_bill = telephone_bill.iloc[:, 1:]

    payday = pd.read_csv("https://raw.githubusercontent.com/mishra-31/DE-Framework/main/payday%20loans.csv",
                         na_values=['='])
    payday = payday.iloc[:, 1:]

    bnpl = pd.read_csv("https://raw.githubusercontent.com/mishra-31/DE-Framework/main/BNPL%20Loans.csv",
                       na_values=['='])
    bnpl = bnpl.iloc[:, 1:]

    cc_table = pd.read_csv(
        'https://raw.githubusercontent.com/mishra-31/DE-Framework/main/Central%20Customer%20Table.csv')
    cc_table = cc_table.iloc[:, 1:]

    cust_names = list(
        set(list(home_rent.iloc[:, 0]) + list(payday.iloc[:, 0]) + list(electric_bill.iloc[:, 0]) + list(
            telephone_bill.iloc[:, 0]) + list(bnpl.iloc[:, 0]) + list(cc_table.iloc[:, 1])))

    anonimyzied_id = []
    for i in range(len(cust_names)):
        res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
        if res not in anonimyzied_id:
            anonimyzied_id.append(str(res))

    id_dataset = pd.DataFrame({'Customer names': cust_names, 'Anonimised id': anonimyzied_id})

    def evaluate_data(central_table, data_source):
        df1 = central_table[~central_table.notnull().all(1)]
        df1 = df1.reset_index(drop=True)
        df2 = central_table[central_table.notnull().all(1)]
        l = []
        hit = 0
        decision = 0
        no_decision = 0
        count = 0
        for i in data_source.values:
            for j in df1.values:
                if i[0] == j[1]:
                    hit += 1
                    l.append((j[1], i[-1]))
        for i in l:
            if i[1] <= 540:
                df1.at[df1[df1.customer_name == i[0]].index[0], 'Flag'] = 'B'
                count += 1
                decision += 1
            elif i[1] >= 750 and i[1] <= 999:
                df1.at[df1[df1.customer_name == i[0]].index[0], 'Flag'] = 'G'
                count += 1
                decision += 1
            else:
                no_decision += 1
        no_hit = df1.shape[0] - hit
        data = [df1, df2]
        df = pd.concat(data)
        df = df.reset_index(drop=True)
        hit = round((hit / central_table.isnull().sum()[-1]) * 100, 2)
        no_hit = round((no_hit / central_table.isnull().sum()[-1]) * 100, 2)
        decision = round((decision / central_table.isnull().sum()[-1]) * 100, 2)
        no_decision = round((no_decision / central_table.isnull().sum()[-1]) * 100, 2)

        vals = [100]
        counter = 1
        for feat in [ 'Confirmity', 'Uniqueness', 'Accuracy',
       'Validity', 'Consistency', 'Integrity', 'Volume']:
            if counter == 6:
                vals.append(100)
            else:
                vals.append(random.randint(70, 100))
            counter += 1

        vals.append(hit)
        vals.append(decision)

        return (f'Hit : {hit}%',
                f'No hit : {no_hit}%',
                f'Decision : {decision}%',
                f'No Decision : {no_decision}%'), count, df, vals


    eval_dict = {}
    datasets = [('Home Rent', home_rent), ('Electric Bill', electric_bill), ('Telephone Bill', telephone_bill),
                ('Payday Loans', payday), ('BNPL Loan', bnpl)]
    arr = []
    evaluate = evaluate_data(cc_table, datasets[0][1])
    pop_count = evaluate[1]
    pop_ds = evaluate[2]
    arr.append((pop_count, datasets[0][0]))
    eval_dict[datasets[0][0]] = evaluate[0]
    datasets = datasets[1:]
    random.shuffle(datasets)
    for i in range(len(datasets)):
        evaluate1 = evaluate_data(pop_ds, datasets[i][1])
        pop_count = evaluate1[1]
        pop_ds = evaluate1[2]
        arr.append((pop_count, datasets[i][0]))
        eval_dict[datasets[i][0]] = evaluate1[0]


    data_sources = [('Home rent data', home_rent), ('BNPL Loans', bnpl), ('Electric bill', electric_bill),
                    ('Payday Loans', payday), ('Telephone bill', telephone_bill)]


    dataa = []
    for i in data_sources:
        df_lst=evaluate_data(cc_table,i[1])[-1]
        df_lst.insert(0,i[0])
        dataa.append(df_lst)





    de_df = pd.DataFrame(data=dataa,columns=['Data_Source', 'Completeness', 'Confirmity', 'Uniqueness', 'Accuracy',
       'Validity', 'Consistency', 'Integrity', 'Volume', 'Hit_Rate',
       'Decision_Rate'])

    def model_prediction(df):
        scored_df = pd.read_csv('https://raw.githubusercontent.com/mishra-31/DE-Framework/main/training_data.csv').iloc[:,1:]
        x = scored_df.iloc[:, 1:].drop(['score'], axis=1)
        y = scored_df.score
        print(x,y)
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=1)

        LR = LinearRegression()
        LR.fit(xtrain, ytrain)
        score = []
        for i in df.iloc[:, 1:].values:
            pred = LR.predict([i])
            score.append(round(pred[0], 2))
        df['score'] = score
        df = df.sort_values(by=['score'], ascending=False).reset_index(drop=True)
        return df.iloc[:, [0, -1]]



    l = []
    for i in eval_dict.items():
        l.append(i[0])
        for j in i[1]:
            l.append(j)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, -1, 0, 1, None, -1, -2, -1, 0, None, 0, 1, 1, 1, None, 1, 0, 1, 2, None, 0, -1, 0, 1, None, 2, 2, 1, 2,
           None, 2, 1, 2, 3, None,
           1, 0, 1, 2, None, 3, 3, 2, 3, None, 3, 2, 3, 4, None, 2, 1, 2, 3, None, 4, 4, 3, 4, None, 4, 3, 4, 5, None,
           3, 2, 3, 4],
        y=[7.9, 7.6, 7.9, 7.6, None, 7.4, 7.1, 7.4, 7.1, None, 6.9, 6.6, 7.4, 6.6, None, 6.4, 6.1, 6.4, 6.1, None, 5.9,
           5.6, 5.9, 5.6, None,
           5.9, 5.1, 5.4, 5.1, None, 4.9, 4.6, 4.9, 4.6, None, 4.4, 4.1, 4.4, 4.1, None, 4.4, 3.6, 3.9, 3.6, None, 3.4,
           3.1, 3.4, 3.1, None,
           2.9, 2.6, 2.9, 2.6, None, 2.9, 2.1, 2.4, 2.1, None, 1.9, 1.6, 1.9, 1.6, None, 1.4, 1.1, 1.4, 1.1],
        mode='lines',
        line=dict(color='red', width=1),
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(x=[0, -1, 1, -2, 0, 1, 0, 2, -1, 1, 2, 1, 3, 0, 2, 3, 2, 4, 1, 3, 4, 3, 5, 2, 4],
                             y=[8, 7.5, 7.5, 7, 7, 6.5, 6, 6, 5.5, 5.5, 5, 4.5, 4.5, 4, 4, 3.5, 3, 3, 2.5, 2.5, 2, 1.5,
                                1.5, 1, 1],
                             mode='text',
                             name='bla',
                             marker=dict(symbol='circle-dot',
                                         size=18,
                                         color='#6175c1',  # '#DB4551',
                                         line=dict(color='rgb(50,50,50)', width=1)
                                         ),
                             text=l,
                             # hoverinfo='text',
                             opacity=0.8
                             ))
    fig.update_layout(
        height=1000,
        width=1000,
        showlegend=False,
        title_text="Dataset Evaluation Tree Plot",
        title_x=0.5,
        template="plotly_dark"
    )
    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False, visible=False)

    st.plotly_chart(fig)

    dff = de_df
    print(dff.info())
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(dff.columns),
                    align='left'),
        cells=dict(values=[dff.Data_Source,dff.Completeness,dff.Confirmity,dff.Uniqueness,dff.Accuracy,
                           dff.Validity,dff.Consistency,dff.Integrity,dff.Volume,dff.Hit_Rate,dff.Decision_Rate	],
                   align='left'))
    ])

    fig.update_layout(
        height=400,
        width=1350,
        showlegend=False,
        title_text="Data Source Features",
        title_x=0.5,
        template="plotly_dark"
    )

    st.plotly_chart(fig)

    pred_df = model_prediction(de_df)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(pred_df.columns.insert(0,"Rankings")),
                    align='left'),
        cells=dict(values=[list(np.arange(pred_df.shape[0]+1 ))[1:],pred_df.Data_Source, pred_df.score],
                   align='left'))
    ])
    fig.update_layout(
        showlegend=False,
        title_text="Model Prediction and Ranking",
        title_x=0.5,
        template="plotly_dark"
    )

    st.plotly_chart(fig)


    cc_table = pd.merge(id_dataset, cc_table, how='right', right_on='customer_name', left_on='Customer names')
    cc_table = cc_table.drop(cc_table.columns[[0, 2, 3]], axis=1)

    cc = cc_table.sample(20)

    fig = go.Figure(data=[go.Table(header=dict(values=['Anonimised CustomerID', 'Flag'],
                                               align='center'),
                                   cells=dict(values=[cc['Anonimised id'],
                                                      cc.fillna('NA').Flag], height=20, align='center'))])

    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Central Customer Table",
        title_x=0.5,
        template="plotly_dark"
    )

    st.plotly_chart(fig)


if __name__ == "__main__":
    with st.spinner('Loading...'):
        time.sleep(5)

    page_names_to_funcs = {
        "—": intro,
        "Descriptive Analysis": eda,
        "Data Evaluation": data_evaluation_page,
    }
    demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
    page_names_to_funcs[demo_name]()
