import streamlit as st 


st.header("Part 1. Display text")

st.caption("st.text(\"Fixed width text\")")
st.text("Fixed width text")

st.caption("st.markdown(\"##  Wallah \")")
st.markdown("##  Wallah ")

st.caption("st.caption(\"Ballons. Hundereds of them...\")")
st.caption("Ballons. Hundereds of them...")

st.caption("st.write(\"most objects\")")
st.write("most objects")

st.caption("st.write(['st','is <','3'])")
st.write(["st","is <","3"])

st.caption('st.title("This is the title of page")')
st.title("This is the title of page")

st.caption('st.header("This is header")')
st.header("This is header")

st.caption('st.subheader("This is suheader")')
st.subheader("This is suheader")

st.caption('''\
        st.code("""\
            for i in range(len(l)):\
                print(i)
        """)''')
st.code("""
        for i in range(len(l)):
            print(i)
        
        """)


st.caption('st.dataframe(data)')
data = {
    "A": [1,2,3,4,5],
    "B": [1,2,3,4,5],
    "C": [1,2,3,4,5],
}
# st.dataframe(data)    


import pandas as pd 
df = pd.DataFrame(data)

st.header("Part 2. Display Data")

st.dataframe(df)

st.caption('st.table(df.iloc[0:3])')
st.table(df.iloc[0:3])


st.caption('st.json(data,expanded=False)')
st.json(data,expanded=False)

st.caption("st.metric(label='temp',value='273 K', delta='1.2 K', delta_color='normal/off/inverse')")
st.metric(label='temp',value='273 K', delta='1.2 K')

st.header("Part 3. Display Media")

st.caption("st.image('path', caption='human image')")
# st.image(r"D:\Ao\Code\MISC\WebCode\Web-Engineering\100-Days-of-Code\Week-1\images\about-1.jpg",caption='Sagar Chhabriya',use_column_width="always")

st.caption('st.audio("data")')
# st.audio(r"C:\Users\Mr Sagar Kumar\Desktop\ReelAudio-24822.mp3")

st.caption('st.video("data")')
# st.video(r"E:\  \Aditya Rikhari\Aditya Rikhari - AANA NAHI (original).mp4")



col1, col2 = st.columns(2)
col1.write("Column1")
col1.write("Column2")


col1, col2, col3 = st.columns([3,1,1])

with col1 :
    st.write("This is column 1")




tab1, tab2 = st.tabs(["Tab1","Tab2"])
tab1.write("This is tab1")
tab2.write("This is tab2")



tab3, tab4, tab5 = st.tabs(["Tab3","Tab4","Tab5"])
tab3.write("This is tab3")
tab4.write("This is tab4")


with tab3:
    st.radio("Select one:",[1,2])


# st.stop()

st.header("Part 4. Control Flow")
st.caption("st.stop()")
st.write("Nothing will be executable after st.stop()")


with st.form(key='my_form'):
    username = st.text_input("Username")
    password = st.text_input("Password")
    st.form_submit_button("Login")



st.header("Part 5. Display Charts")

st.caption("st.line_chart([1,2,3,4,5])")
st.line_chart([1,2,3,4,5])

st.caption("st.area_chart([1,2,3,4,5])")
st.area_chart([1,2,3,4,5])

st.caption("st.bar_chart([1,2,3,4,5])")
st.bar_chart([1,2,3,4,5])


st.caption("st.altair_chart(), Bit complicated")
import altair as alt
d = pd.DataFrame({
    'x': [1, 2, 3, 4],
    'y': [4, 3, 2, 1]
})

# Try different mark_methods()
chart = alt.Chart(d).mark_rect().encode(x='x',y='y')
st.altair_chart(chart,use_container_width=True)


# 5.1 Lineplot
st.subheader("5.1 Lineplot")

st.code("""
# Creating a simple DataFrame
data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10),
    'value': [10, 20, 15, 30, 40, 35, 50, 60, 55, 70]
})

# Set date as index
data.set_index('date', inplace=True)

# Display line chart
st.line_chart(data)
""")

# 5.2 Bar Chart
data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10),
    'value': [10, 20, 15, 30, 40, 35, 50, 60, 55, 70]
})
data.set_index('date', inplace=True)
st.line_chart(data)



st.subheader("5.2 Bar Chart")

st.code("""
data = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D', 'E'],
    'values': [10, 20, 30, 40, 50]
})

# Display bar chart
st.bar_chart(data.set_index('category'))
""")

# Display the chart
data = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D', 'E'],
    'values': [10, 20, 30, 40, 50]
})
st.bar_chart(data.set_index('category'))



# 5.3 Area Chart
st.code("""
data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10),
    'value':[10,20,15,30,40,35,50,60,55,70]
})

# Set date as index
data.set_index('date',inplace=True)

# Display area chart
st.area_chart(data)
        
""")


data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=10),
    'value': [10,20,15,30,40,35,50,60,55,70]
})

data.set_index('date',inplace=True)
st.area_chart(data)


# 5.4 Scatter Plot
st.subheader("5.4 Scatter Plot")
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.code('''
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(100)
y = np.random.randn(100)

fig, ax = plt.subplots()
ax.scatter(x,y)

st.pyplot(fig)


''')
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)

fig, ax = plt.subplots()
ax.scatter(x,y)

st.pyplot(fig)


# 5.5 Bar Chart
st.subheader("5.5 Bar Chart")
st.code('''
import plotly.express as px
data = {
    'category':['A','B','C','D','E'],
    'value':[10,20,30,40,50]
}

fig = px.bar(data, x='category',y='value',title='Bar Chart Example')
st.plotly_chart(fig)        

''')

import plotly.express as px
data = {
    'category':['A','B','C','D','E'],
    'value':[10,20,30,40,50]
}

fig = px.bar(data, x='category',y='value',title='Bar Chart Example')
st.plotly_chart(fig)


# 5.6 Pie Chart
st.subheader("5.6 Pie Chart")

st.code('''
fig = px.pie(data, names='category',values='value', title='Pie Chart Example')
st.plotly_chart(fig)
''')


fig = px.pie(data, names='category',values='value', title='Pie Chart Example')
st.plotly_chart(fig)


# 5.7 Histogram
st.subheader("5.7 Histogram")

import altair as alt
data = pd.DataFrame({
     'values': [10, 20, 15, 30, 40, 35, 50, 60, 55, 70, 45, 60]
})


chart = alt.Chart(data).mark_bar().encode(
    alt.X('Values:Q', bin=True),
    alt.Y('count():Q')
)
st.altair_chart(chart, use_container_width=True)

# 5.8  Map
st.subheader("Map")
st.code("""
data = pd.DataFrame({
    'lat': [37.7749, 34.0522, 40.7128],
    'lon':[-122.4194, -118.2437, -74.0060]
})
st.map(data)
""")

data = pd.DataFrame({
    'lat': [37.7749, 34.0522, 40.7128],
    'lon':[-122.4194, -118.2437, -74.0060]
})
st.map(data)


# 9. HeatMap
st.subheader("Heat Map")

st.code('''
z = np.random.randn(10,10)
fig = px.imshow(z, color_continuous_scale='Viridis', title="Heatmap Example")
st.plotly_chart(fig)        
''')


z = np.random.randn(10,10)
fig = px.imshow(z, color_continuous_scale='Viridis', title="Heatmap Example")
st.plotly_chart(fig)


