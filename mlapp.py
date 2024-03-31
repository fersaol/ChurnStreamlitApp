import streamlit as st
import numpy as np
from general_purpose import ml_model_manager
from time import sleep


def app():

    st.title("Churn Prediction")
    st.header("Scikit-Learn Machine Learning Model")

    def RoboFer():
        with st.chat_message(
            name="assistant",
            avatar="assistant"):
            st.write("""Hi!! 👋, This is a machine learning model that 
                     predicts whether a customer will churn or not.""")
            user_name = st.chat_input("What is your name?")

        if user_name != None:
            st.chat_message(name="assistant").write(f"Nice to meet you, {user_name}.")
            sleep(1)

            with st.chat_message(name="assistant"):
                st.write("""
                         to get started place the values in the left pane 
                         clicking the > arrow at the top left corner and 
                         then press submit""")
                sleep(1)

            st.chat_message(name="assistant").write("then, you will get a prediction. I hope you like!! 😃")

    features = st.sidebar.form(key='features',
                               clear_on_submit=True)

    
    with features:
        # include the link to the Dash Exploratory data analysis
        # st.page_link(page="pages/app.py",
                     #label="Explore the Data",
                     #use_container_width=True)

        age = st.number_input("Age",
                                   min_value=1,
                                   max_value=120,
                                   value=1)
        
        active = st.selectbox("Is an active member?",
                                options=[True,False])
        
        num_products = st.number_input("Number of Products", 
                                   min_value=1, 
                                   max_value=50, 
                                   value=1)
        
        balance = st.number_input("Balance",0)

        credit_card = st.selectbox("Has it Credit Card?",
                                options=[True,False])
        
 
        if age >= 1 and age <= 120:

            submit = st.form_submit_button('Predict', 
                                use_container_width=True,
                                type="primary")
        else:
            submit = st.form_submit_button('Predict',
                                        type="primary",
                                        disabled=True,
                                        use_container_width=True)


    if submit:

        model = ml_model_manager(
            name="ChurningGradientBoostingClassifier_v0",
            save=False)
        
        prediction = model.predict([[balance,np.log(age),num_products, credit_card,active]])
    
        data = {
            'Age': age,
            'Active Client': active,
            'Number of Products': num_products,
            'balance': balance,
            'Credit Card': credit_card,
        }

        if prediction == 1:

            st.warning('Customer will churn',icon="🚨")

            left_column, right_column = st.columns((0.7,0.3))

            left_column.image(image="images/willchurn.png",
                     width=400)
            
            right_column.text("Relevant data:")
            right_column.dataframe(data)

            st.subheader("please, call the customer and make sure he is alright")
            
            
        elif prediction == 0:
            st.success('Customer will not churn')
            st.image(image="images/calmSea.png",width=300,use_column_width=1)
            st.balloons()
    else:
        RoboFer()



if __name__ == "__main__":
    app()
