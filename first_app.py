import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from time import time
import streamlit.components.v1 as components
st.set_option('deprecation.showPyplotGlobalUse', False)

# # start_time = time()
# # data = pd.read_csv('C:/Users/e.gadimov/Desktop/data_17.csv')
# # end_time = time()
# # print(end_time-start_time)
# # data.shape


st.write("""
# Gözləmə müddətinin təyin olunması 
""")


# st.write("Here's our first attempt at using data to create a table:")

# chart_data = pd.DataFrame(
#      np.random.randn(20, 2),
#      columns=['a', 'b'])

# st.line_chart(chart_data)

# # st.write(data.head())



# DATE_COLUMN = 'date/time'
DATA_URL = ('C:/Users/e.gadimov/Desktop/ddd.csv')


@st.cache
def load_data(nrows):
    # data = pd.read_csv(DATA_URL, nrows=nrows)
    data = pd.read_csv(DATA_URL)
    # lowercase = lambda x: str(x).lower()
    # data.rename(lowercase, axis='columns', inplace=True)
    # data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(300000)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')


sorted_branch = sorted(data['branch_name'].unique())

st.markdown("### **Select Branch:**")
select_branch = []
select_branch.append(st.selectbox('', sorted_branch))



sorted_days = sorted(data['date'].unique())

st.markdown("### **Select Date:**")
select_date = []
select_date.append(st.selectbox('', sorted_days))


sorted_roots = sorted(data['organization_name'].unique())

st.markdown("### **Select Organization:**")
select_root = []
select_root.append(st.selectbox('', sorted_roots))

#############################################################

#Filter df based on selection
branch_df = data[ data['branch_name'].isin(select_branch) 
            & data['date'].isin(select_date) 
            & data['organization_name'].isin(select_root) ]




# st.subheader('Number of pickups by hour')

# hist_values = np.histogram(
#     data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]


# st.bar_chart(hist_values)

## Matplotlib example


with st.echo(code_location='below'):
    import plotly.express as px

    # color_discrete_map = {'Hazır sənəd':'red', 
    #                       'Sənəd Qəbulu':'blue',
    #                       'İnformasiya Xidməti':'green'}

    fig = px.scatter(
        x=branch_df["predicted_wait_time"],
        y= branch_df["waiting_time_min"],
        color=branch_df['letter'],
        width=800, height=800,
        # color_discrete_map=color_discrete_map
    )

    fig.update_layout(
        xaxis_title="Predicted Wait Time",
        yaxis_title="Real Wait Time",
    )

    st.write(fig)


# with st.echo(code_location='below'):
#     import seaborn as sns

#     # fig, ax = plt.subplots()
#     sns.jointplot(
#         x = data["predicted_wait_time"],
#         y = data["waiting_time_min"],
#         height=10,
#         kind='reg',
#         xlim=(0, 500),
#         ylim=(0, 500),
        
#     )

#     # fig.update_layout(
#     #     xaxis_title="Predicted Wait Time",
#     #     yaxis_title="Real Wait Time",
#     # )

#     st.pyplot()