import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go 
import streamlit as st

df = pd.read_excel('Failure_points.xlsx')
st.title('Failure points for Businesses')
st.write(df.head())

st.title('Bar-Chart: Failure points Percentages for Businesses')
fig = px.bar(df, x="Percent", y="Failure point", color='Percent', width=950, height=850, 
             template="plotly_white", color_continuous_scale = "Spectral_r", title='Business Failure Reasons')
fig.update_yaxes(ticks="",showgrid=False, zeroline=True)
fig.update_xaxes(ticks="",showgrid=False, zeroline=True)
fig.update_xaxes(showspikes=True, spikecolor="pink", spikethickness=1)#spikesnap="cursor", spikemode="across")
fig.update_yaxes(showspikes=True, spikecolor="orange", spikethickness=1)
fig.update_layout(spikedistance=1000, hoverdistance=1000)
fig.update_layout(hovermode="y", 
                  hoverlabel=dict(#bgcolor="white",
                                  font_size=14,
                                  font_family="Rockwell"
                                 ))
fig.show()
st.pyplot(fig)

st.title('Sunburst diagram: Failure points in a Business area')
fig = px.sunburst(df, path=['Reasons', 'Failure point'], values='Percent', width=1000, height=900, 
             template="plotly_white",color='Reasons', title='Failure points in a Business area')
fig.update_traces(textfont_size=14,textfont_color="white",leaf=dict(opacity=0.75))
fig.update_layout(hoverlabel=dict(bgcolor="white",
                                  font_size=14,
                                  font_family="Rockwell"
                                 ))
fig.show()
st.pyplot(fig)

df0 = pd.ExcelFile("Failure_points1.xlsx")
df1 = pd.read_excel(df0, 'Sheet1')
df2 = pd.read_excel(df0, 'Sheet2')
df3 = pd.read_excel(df0, 'Sheet3')
df4 = pd.read_excel(df0, 'Sheet4')
df5 = pd.read_excel(df0, 'Sheet5')

st.title('Failure points related to the Business')
fig1 = px.sunburst(df1, path=['Reasons', 'Failure point'], values='Percentage', width=900, height=900, 
             template="plotly_white", color_continuous_scale = "reds", color='Percentage', 
                   title='Failure points related to the Business')
fig1.update_traces(textfont_size=14,textfont_color="black",leaf=dict(opacity=0.8))
fig1.update_layout(hoverlabel=dict(bgcolor="white",
                                  font_size=14,
                                  font_family="Rockwell"
                                 ))
fig1.show()
st.pyplot(fig1)

st.title('Failure points related to the People')
fig2 = px.sunburst(df2, path=['Reasons', 'Failure point'], values='Percentage', width=900, height=900, 
             template="plotly_white", color_continuous_scale = "oranges", color='Percentage',
                  title='Failure points related to the People')
fig2.update_traces(textfont_size=14,textfont_color="black",leaf=dict(opacity=0.8))
fig2.update_layout(hoverlabel=dict(bgcolor="white",
                                  font_size=14,
                                  font_family="Rockwell"
                                 ))
fig2.show()
st.pyplot(fig2)

st.title('Failure points related to the Environment')
fig3 = px.sunburst(df3, path=['Reasons', 'Failure point'], values='Percentage', width=900, height=900, 
             template="plotly_white", color_continuous_scale = "purples", color='Percentage',
                  title='Failure points related to the Environment')
fig3.update_traces(textfont_size=14,textfont_color="black",leaf=dict(opacity=0.8))
fig3.update_layout(hoverlabel=dict(bgcolor="white",
                                  font_size=14,
                                  font_family="Rockwell"
                                 ))
fig3.show()
st.pyplot(fig3)

st.title('Failure points related to the Product')
fig4 = px.sunburst(df4, path=['Reasons', 'Failure point'], values='Percentage', width=900, height=900, 
             template="plotly_white", color_continuous_scale = "blues", color='Percentage',
                  title='Failure points related to the Product')
fig4.update_traces(textfont_size=14,textfont_color="black",leaf=dict(opacity=0.8))
fig4.update_layout(hoverlabel=dict(bgcolor="white",
                                  font_size=14,
                                  font_family="Rockwell"
                                 ))
fig4.show()
st.pyplot(fig4)

st.title('Failure points related to the Customer/User')
fig5 = px.sunburst(df5, path=['Reasons', 'Failure point'], values='Percentage', width=900, height=900, 
             template="plotly_white", color_continuous_scale = "greens", color='Percentage',
                  title='Failure points related to the Customer/User')
fig5.update_traces(textfont_size=14,textfont_color="black",leaf=dict(opacity=0.8))
fig5.update_layout(hoverlabel=dict(bgcolor="white",
                                  font_size=14,
                                  font_family="Rockwell"
                                 ))
fig5.show()
st.pyplot(fig5)


Reasons=df['Failure point']
Percent=df['Percent']
fig = go.Figure(data=[go.Pie(labels=Reasons, values=Percent, insidetextorientation='radial', 
                            title='Failure points in decreasing order Pie-chart', titlefont_size=20)])
fig.update_traces(hoverinfo='label+value', textinfo='label', textfont_size=12,
                  marker=dict(colors=Percent), showlegend=False, )
fig.update_layout(hovermode="y", height=950, width=1000,
                  hoverlabel=dict(bgcolor="white",
                                  font_size=16,
                                  font_family="Rockwell"
                                 ))
fig.show()
st.pyplot(fig)