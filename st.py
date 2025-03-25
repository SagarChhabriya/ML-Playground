import streamlit as st 

# StreamlitAPIException: set_page_config() can only be called once per app page, 
# and must be called as the first Streamlit command in your script.
st.set_page_config(layout='centered')

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



with st.echo():
    col1, col2 = st.columns(2)
    col1.write("Column1")
    col1.write("Column2")




with st.echo():
    col1, col2, col3 = st.columns([3,1,1])

    with col1 :
        st.write("This is column 1")
    with col2 :
        st.write("This is column 2")
    with col3 :
        st.write("This is column 3")



with st.echo():
    tab1, tab2 = st.tabs(["Tab1","Tab2"])
    tab1.write("This is tab1")
    tab2.write("This is tab2")

with st.echo():
    tab3, tab4, tab5 = st.tabs(["Tab3","Tab4","Tab5"])
    tab3.write("This is tab3")
    tab4.write("This is tab4")

    with tab3:
        st.radio("Select one:",[1,2])


# st.stop()

st.header("Part 4. Control Flow")
st.caption("st.stop()")
st.write("Nothing will be executable after st.stop()")


with st.echo():
    with st.form(key='my_form'):
        username = st.text_input("Username")
        password = st.text_input("Password")
        st.form_submit_button("Login")



st.header("Part 5. Display Charts")
# 5.1 Lineplot
st.subheader("5.1 Lineplot")

st.code('st.line_chart([1,2,3,4,5])')
st.line_chart([1,2,3,4,5])


with st.echo():

    # 5.2 Bar Chart
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10),
        'value': [10, 20, 15, 30, 40, 35, 50, 60, 55, 70]
    })
    data.set_index('date', inplace=True)
    st.line_chart(data)



st.subheader("5.2 Bar Chart")


st.code('st.bar_chart([1,2,3,4,5])')
st.bar_chart([1,2,3,4,5])


with st.echo():
    # Display the chart
    data = pd.DataFrame({
        'category': ['A', 'B', 'C', 'D', 'E'],
        'values': [10, 20, 30, 40, 50]
    })
    st.bar_chart(data.set_index('category'))



# 5.3 Area Chart

st.code('st.area_chart([1,3,5,7,9])')
st.area_chart([1,3,5,7,9])

with st.echo():
    data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10),
        'value': [10,20,15,30,40,35,50,60,55,70]
    })

    data.set_index('date',inplace=True)
    st.area_chart(data)


# 5.4 Scatter Plot
st.subheader("5.4 Scatter Plot")

with st.echo():
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np

    np.random.seed(42)
    x = np.random.randn(100)
    y = np.random.randn(100)

    fig, ax = plt.subplots()
    ax.scatter(x,y)

    st.pyplot(fig)


# 5.5 Bar Chart
st.subheader("5.5 Bar Chart")

with st.echo():
    import plotly.express as px
    data = {
        'category':['A','B','C','D','E'],
        'value':[10,20,30,40,50]
    }

    fig = px.bar(data, x='category',y='value',title='Bar Chart Example')
    st.plotly_chart(fig)


# 5.6 Pie Chart
st.subheader("5.6 Pie Chart")

with st.echo():
    fig = px.pie(data, names='category',values='value', title='Pie Chart Example')
    st.plotly_chart(fig)


# 5.7 Histogram
st.subheader("5.7 Histogram")

with st.echo():
    import altair as alt
    data = pd.DataFrame({
        'values': [10, 20, 15, 30, 40, 35, 50, 60, 55, 70, 45, 60]
    })


    chart = alt.Chart(data).mark_bar().encode(
        alt.X('values:Q', bin=True),
        alt.Y('count():Q')
    )
    st.altair_chart(chart, use_container_width=True)

# Example 2

with st.echo():
    import altair as alt
    d = pd.DataFrame({
        'x': [1, 2, 3, 4],
        'y': [4, 3, 2, 1]
    })

# Try different mark_methods()
chart = alt.Chart(d).mark_rect().encode(x='x',y='y')
st.altair_chart(chart,use_container_width=True)






# 5.8  Map
st.subheader("5.8 Map")

with st.echo():
    data = pd.DataFrame({
        'lat': [37.7749, 34.0522, 40.7128],
        'lon':[-122.4194, -118.2437, -74.0060]
    })
    st.map(data)


# 9. HeatMap
st.subheader("Heat Map")

with st.echo():
    z = np.random.randn(10,10)
    fig = px.imshow(z, color_continuous_scale='Viridis', title="Heatmap Example")
    st.plotly_chart(fig)



##################################################################################

st.header("Part 6: Display Interactive Widgets")

st.code("st.button(\"Don\'t touch me!\")")
st.button("Don't touch me!")

st.code('st.checkbox("Check me out!")')
st.checkbox("Check me out!")

st.code('st.radio(\'Radio\',[1,2,4])')
st.radio('Radio',[1,2,4])

st.code('st.selectbox(\'Select\',[1,2,4])')
st.selectbox('Select',[1,2,4])

st.code('st.multiselect(\'Multiselect\',["Apple","Banana","Orange"])')
st.multiselect('Multiselect',["Apple","Banana","Orange"])

st.code('st.slider(\'Slide Me\',min_value=0, max_value=10)')
st.slider('Slide Me',min_value=0, max_value=10)

st.code('st.text_input(\'Enter some text\')')
st.text_input('Enter some text')


st.code('st.number_input(\'Enter a number\')')
st.number_input('Enter a number')

st.code("st.text_area('Area for textual entry')")
st.text_area('Area for textual entry')



with st.echo():
    st.date_input('Date input')

with st.echo():
    st.time_input('Time Entry')

with st.echo():
    st.file_uploader('File uploader')


# st.balloons()
with st.echo():
    for i in range(int(st.number_input('Num:'))): st.write(i)

with st.echo():
    if st.sidebar.selectbox('I:',['f','g']) == 'f': st.write('ok')

with st.echo():
    st.slider('Number',1,100)


st.subheader('Display Code')

with st.echo():
    with st.echo():
        # Code below both executed and printed
        foo ='bar'
        st.write(foo)
    
    
st.header('6. Display Progress and Status')

with st.echo():
    st.progress(text='85%', value=85)

with st.echo():
    with st.spinner(text='In Progress'):
        import time 
        time.sleep(2)
        st.success('Done')

with st.echo():
    st.error('Error Message')

with st.echo():
    st.warning('Warning Message')

with st.echo():
    st.info('Info Message')

with st.echo():
    st.exception("Is this error, wow, I didn't know!!")



st.header('7. Placeholders, help, and options')

with st.echo():
    place_holder = st.empty()
    place_holder.text('Replaced')


with st.echo():
    st.help(pd.DataFrame)

# st.get_option('key')
# st.set_option('key')



with st.echo():
    table = st.table(data)
    table.add_rows(data)

with st.echo():
    chart = st.line_chart(data)
    chart.add_rows(data)


# st.chat_input
# st.experimental_audio_input


# st.cache
# st.cache_data
# st.cache_resource
# st.camera_input
# st.chat_message
# st.color_picker
# st.column_config
# st.connection
# st.container
# st.container

# st.data_editor
# st.dialog
# st.divider
# st.download_button


# st.echo
# st.empty
# st.error
# st.exception
# st.expander


# st.feedback
# st.file_uploader
# st.form
# st.form_submit_button
# st.fragment


# st.get_option
# st.graphviz_chart


# st.header
# st.help
# st.html

# st.image
# st.info


# st.json

# st.latex
# st.line_chart
# st.line_chart
# st.login
# st.logo
# st.logout
# st.vega_lite_chart


# st.map
# st.markdown
# st.metric
# st.multiselect



# st.navigation
# st.number_input

# st.get_option
# st.set_option

# st.page_link
# st.pills
# st.plotly_chart
# st.popover
# st.progress
# st.pydeck_chart
# st.pyplot
# st.Page
# st.set_page_config


# st.query_params

# st.radio
# st.rerun

# st.scatter_chart
# st.secrets
# st.segmented_control
# st.select_slider
# st.selectbox
# st.session_state
# st.set_option
# st.set_page_config
# st.sidebar
# st.slider
# st.snow
# st.spinner
# st.switch_page
# st.status
# st.stop
# st.subheader
# st.success
# st.form_submit_button
# st.write_stream



# st.table
# st.tabs
# st.text
# st.text_area
# st.text_input
# st.time_input
# st.title
# st.toast
# st.toggle


# st._update_logger


# st.vega_lite_chart
# st.video

# st.warning
# st.write
# st.write_stream


