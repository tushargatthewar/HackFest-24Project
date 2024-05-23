from io import BytesIO
import zipfile
from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for
from flask import make_response

import dash
from dash import dcc, html
import plotly.graph_objs as go
import pymongo
import dash_bootstrap_components as dbc
from pymongo import MongoClient
from flask_session import Session
from flask import session
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter  
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import numpy as np
from datetime import datetime
import boto3
import csv
import google.generativeai as genai
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import google.generativeai as genai
import textwrap
from reportlab.platypus import KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.platypus.flowables import KeepTogether
import os
import requests
import boto3

import torch
import pandas as pd

# Initialize Flask app
app_flask = Flask(__name__)
app_flask.secret_key = 'your_secret_key_here'
app = dash.Dash(__name__)

region_name = 'ap-south-1'


translate_client = boto3.client('translate', region_name=region_name)
comprehend_client = boto3.client('comprehend', region_name=region_name)

API_KEY = "AIzaSyCdrUBlS3L2H-p7Toe5YDYFtSqEfRvjf7k"


# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['hackfest']
collection = db['vaccination1']
collection1 = db['transportation']
collection2 = db['asha_worker_register']
collection4=db['nutritionForm']
collection5=db['Pwomen']
collection7 = db['Gscheme']

# Global variables
accessibility_values = []
num_members_vaccinated_values = []




genai.configure(api_key=API_KEY)

# Instantiate the model
model = genai.GenerativeModel('gemini-pro')

def translate_text(text, source_lang='auto', target_lang='en'):
    response = translate_client.translate_text(Text=text, SourceLanguageCode=source_lang,
                                               TargetLanguageCode=target_lang)
    translated_text = response['TranslatedText']
    return translated_text

def analyze_sentiment(text):
    if not text.strip():  # Skip empty feedbacks
        return None
    response = comprehend_client.detect_sentiment(Text=text, LanguageCode='en')
    sentiment = response['Sentiment']
    return sentiment

def process_feedbacks(csv_file):
    scheme_feedbacks = {}
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            scheme = row['scheme']
            feedback = row['feedback']
            if scheme not in scheme_feedbacks:
                scheme_feedbacks[scheme] = []
            scheme_feedbacks[scheme].append(feedback)
    return scheme_feedbacks

def sort_negative_feedbacks(scheme_feedbacks):
    scheme_negative_feedbacks = {}
    for scheme, feedbacks in scheme_feedbacks.items():
        negative_feedbacks = []
        for feedback in feedbacks:
            sentiment = analyze_sentiment(feedback)
            if sentiment == 'NEGATIVE':
                negative_feedbacks.append(feedback)
        if negative_feedbacks:  # Only store if there are negative feedbacks
            scheme_negative_feedbacks[scheme] = negative_feedbacks
    return scheme_negative_feedbacks

def generate_content(input_text, max_length=200):
    response = model.generate_content(input_text)
    generated_text = response.text
    # Truncate the text if it exceeds the maximum length
    if len(generated_text) > max_length:
        generated_text = generated_text[:max_length]
    return generated_text

def generate_pdf(output):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    y = 750
    for line in output.split('\n'):
        c.drawString(50, y, line)
        y -= 20
    c.save()
    buffer.seek(0)
    return buffer

@app_flask.route('/feedback')
def feedback():
    return render_template('Feedback.html')

@app_flask.route('/process_feedbacks', methods=['POST'])
def process_feedbacks_route():
    csv_file = 'hackfest.feedback.csv'  # Provide the path to your CSV file
    scheme_feedbacks = process_feedbacks(csv_file)
    scheme_negative_feedbacks = sort_negative_feedbacks(scheme_feedbacks)
    output = ""
    for scheme, feedbacks in scheme_feedbacks.items():
        output += f"Scheme: {scheme}\n"
        output += f"Negative Feedbacks: {feedbacks}\n"

        # Check if negative feedbacks exist for the current scheme
        if scheme in scheme_negative_feedbacks:
            all_negative_feedbacks = "\n".join(scheme_negative_feedbacks[scheme])
            # Translate all negative feedbacks to English
            translated_feedbacks = translate_text(all_negative_feedbacks, source_lang='auto', target_lang='en')
            # Generate content using Gemini AI for all negative feedbacks
            generated_content = generate_content(translated_feedbacks)
            # Display generated content
            output += "Explanation of Negative Feedbacks:\n"
            output += textwrap.fill(generated_content, width=80) + "\n"

            # Generate solutions for all negative feedbacks
            output += "\nSolutions:\n"
            for feedback in scheme_negative_feedbacks[scheme]:
                translated_feedback = translate_text(feedback, source_lang='auto', target_lang='en')
                generated_solution = generate_content(translated_feedback)
                output += textwrap.fill(generated_solution, width=80) + "\n"
        else:
            output += "No negative feedbacks found for this scheme.\n"

        output += "\n\n"

    # Generate PDF
    pdf_buffer = generate_pdf(output)

    # Create response
    response = make_response(pdf_buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=output.pdf'
    return response


# Initialize Dash app within Flask app
app_dash = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard/')
app_dash2 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard1/')
app_dash3 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard2/')
app_dash4 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard3/')


# traportaion dahhses
app_dash6 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard4/')
app_dash7 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard5/')
app_dash8 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard6/')
app_dash9 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard7/')



app_dash10 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard8/')
app_dash12 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard10/')
app_dash11 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard9/')
app_dash13 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard11/')
app_dash14 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard12/')
app_dash15 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard13/')
app_dash16 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard14/')
app_dash17 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard15/')
app_dash18 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard16/')
app_dash19 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard17/')




app_dash15.layout = html.Div(children=[
    dcc.Graph(
        id='income-children-milk-scatter-plot'
    )
])


# Define layout for Dash app
app_dash.layout = html.Div(children=[
    dcc.Graph(
        id='donut-chart'
    )
])

# Define layout for Dash app2
app_dash2.layout = html.Div(children=[
    dcc.Graph(
        id='donut-chart2'
    )
])
# Define layout for Dash app3
app_dash3.layout = html.Div(children=[
    dcc.Graph(
        id='bar-graph'
    )
])
app_dash4.layout = html.Div(children=[
    dcc.Graph(
        id='line-graph'
    )
])

app_dash6.layout = html.Div(children=[
    dcc.Graph(
        id='income-transportation-spending'
      
    )
])
app_dash7.layout = html.Div(children=[
    dcc.Graph(
        id='urban-rural-visits-bar-chart'
      
    )
])

app_dash8.layout = html.Div(children=[
    dcc.Graph(
        id='fees-per-visit'
      
    )
])
app_dash9.layout = html.Div(children=[
    dcc.Graph(
        id='hospital-type-bar-chart'
      
    )
])

app_dash12.layout = html.Div(children=[
    dcc.Graph(
        id='primary-food-source-affordability-horizontal-bar-chart'
    )
])
app_dash13.layout = html.Div(children=[
    dcc.Graph(
        id='household-size-daily-food-consumption-bar-chart'
    )
])

app_dash14.layout = html.Div(children=[
    dcc.Graph(
        id='clean-water-water-source-donut-chart'
    )
])



app_dash16.layout = html.Div(children=[
    dcc.Graph(
        id='income-children-milk-scatter-plot'
    )
])

app_dash17.layout = html.Div(children=[
    dcc.Graph(
        id='financial-barriers-government-schemes-donut-chart'
    )
])

app_dash18.layout = html.Div(children=[
    dcc.Graph(
        id='delivery-preference-donut-chart'
    )
])
app_dash19.layout = html.Div(children=[
    dcc.Graph(
        id='healthcare-distance-box-plot'
    )
])



# Define callback to update graph dynamically
@app_dash.callback(
    dash.dependencies.Output('donut-chart', 'figure'),
    [dash.dependencies.Input('donut-chart', 'id')]
)
def update_donut_chart(_):
    # Fetch latest values of accessibility and num_members_vaccinated
    village_name = session.get('village_name', None)
    village_data = list(collection.find({"village_name": village_name}))

    # Initialize counters
    accessibility_yes_members = []
    total_members_vaccinated = []

    
    for doc in village_data:
        if doc.get('accessibility') == 1:
            accessibility_yes_members.append(doc.get('num_members_vaccinated', 0))
        total_members_vaccinated.append(doc.get('num_members_vaccinated', 0))

   
    bar_data = [
        go.Bar(
            x=['Accessibility Yes'],
            y=[sum(accessibility_yes_members)],
            name='Accessibility Yes',
            marker=dict(color='#1bcfb4')
        ),
        go.Bar(
            x=['Total Members Vaccinated'],
            y=[sum(total_members_vaccinated)],
            name='Total Members Vaccinated',
            marker=dict(color='#a05aff')
        )
    ]

    # Set the layout for the bar chart
    layout = {
        'barmode': 'group',
        'title': 'Number of Vaccinated Members by Accessibility',
        'xaxis': {'title': 'Category'},
        'yaxis': {'title': 'Number of Vaccinated Members'}
    }

    return {'data': bar_data, 'layout': layout}


  



# Define callback to update graph dynamically for app_dash2
@app_dash2.callback(
    dash.dependencies.Output('donut-chart2', 'figure'),
    [dash.dependencies.Input('donut-chart2', 'id')]
)
def update_graph2(_):
    # Extract relevant data for visualization in app_dash2
    village_name = session.get('village_name', None)
    data_from_mongodb = list(collection.find({"village_name": village_name}))
    accessibility_values = [doc['accessibility'] for doc in data_from_mongodb]
    total_accessible = sum(accessibility_values)
   
    return {
        'data': [
            go.Pie(
                labels=['Accessibility Available', 'Accessibility Not Available'],
                values=[total_accessible, len(data_from_mongodb) - total_accessible],
                hole=0.4,
                marker=dict(colors=['#007bff', '#6f42c1'])
            )
        ],
        'layout': {
            'title': 'Relationship between Community Engagement and Accessibility'
        }
    }


@app_dash3.callback(
    dash.dependencies.Output('bar-graph', 'figure'),
    [dash.dependencies.Input('bar-graph', 'id')]
)
def update_graph3(_):
    # Fetch data from the MongoDB collection for Asha Worker Help
    village_name = session.get('village_name', None)
    data_from_mongodb = list(collection.find({"village_name": village_name}))
    asha_worker_help_values = [doc.get('asha_worker_help', 0) for doc in data_from_mongodb]
    asha_worker_help_yes = sum(1 for val in asha_worker_help_values if val == 1)
    asha_worker_help_no = sum(1 for val in asha_worker_help_values if val == 0)
    

    return {
        'data': [
            go.Bar(
                x=['Yes', 'No'],
                y=[asha_worker_help_yes, asha_worker_help_no],
                marker=dict(color=['#8E44AD', '#00cccc'])
            )
        ],
        'layout': {
            'title': 'Relationship between Asha Worker Help and Total Family Members',
            'xaxis': {'title': 'Asha Worker Help'},
            'yaxis': {'title': 'Count'}
        }
    }

@app_dash4.callback(
    dash.dependencies.Output('line-graph', 'figure'),
    [dash.dependencies.Input('line-graph', 'id')]
)
def update_graph4(_):
    # Fetch data from the MongoDB collection for Awareness and Members Vaccinated
    village_name = session.get('village_name', None)
    data_from_mongodb = list(collection.find({"village_name": village_name}))
    awareness_values = [doc.get('awareness', 0) for doc in data_from_mongodb]
    num_members_vaccinated_values = [doc.get('num_members_vaccinated', 0) for doc in data_from_mongodb]

    
    line_color = '#1f77b4'  # Blue
    marker_color = '#ff7f0e'  # Orange

    return {
        'data': [
            go.Scatter(
                x=awareness_values,
                y=num_members_vaccinated_values,
                mode='lines+markers',
                line=dict(color=line_color),  # Set line color
                marker=dict(color=marker_color)  # Set marker color
            )
        ],
        'layout': {
            'title': 'Number of Members Vaccinated Over Time',
            'xaxis': {'title': 'Awareness (0: No, 1: Yes)'},
            'yaxis': {'title': 'Number of Members Vaccinated'}
        }
    }

@app_dash6.callback(
    dash.dependencies.Output('income-transportation-spending', 'figure'),
    [dash.dependencies.Input('income-transportation-spending', 'id')]
)
def update_graph6(_):
    # Fetch data from MongoDB collection for Income vs. Transportation Spending
    village_name = session.get('village_name', None)
    data_from_mongodb1 = list(collection1.find({"village_name": village_name}))
    
    # Update the graph with new data
    figure = {
        'data': [
            go.Scatter(
                x=[data['income'] if 'income' in data else 0 for data in data_from_mongodb1],
                y=[data['transportation_spending'] if 'transportation_spending' in data else 0 for data in data_from_mongodb1],
                mode='markers',
                marker=dict(color='#007bff')  # Adjust marker color if needed
            )
        ],
        'layout': {
            'title': 'Income vs. Transportation Spending',
            'xaxis': {'title': 'Income'},
            'yaxis': {'title': 'Transportation Spending'}
        }
    }
    
    return figure

@app_dash7.callback(
    dash.dependencies.Output('urban-rural-visits-bar-chart', 'figure'),
    [dash.dependencies.Input('urban-rural-visits-bar-chart', 'id')]
)
def update_graph(_):
    # Assuming you have data available to update the graph dynamically
    # Replace the following lines with your actual data retrieval and processing logic
    village_name = session.get('village_name', None)
    data_from_mongodb1 = list(collection1.find({"village_name": village_name}))
    urban_visits = sum(int(data.get('urban_visits', '0')) for data in data_from_mongodb1)
    rural_visits = sum(int(data.get('rural_visits', '0')) for data in data_from_mongodb1)
    return {
        'data': [
            go.Bar(
                x=['Urban Visits', 'Rural Visits'],
                y=[urban_visits, rural_visits],
                marker=dict(color=['#03A9F4', '#66FFCC'])
            )
        ],
        'layout': {
            'title': 'Urban vs. Rural Visits',
            'xaxis': {'title': 'Type of Visit'},
            'yaxis': {'title': 'Number of Visits'}
        }
    }

@app_dash8.callback(
    dash.dependencies.Output('fees-per-visit', 'figure'),
    [dash.dependencies.Input('fees-per-visit', 'id')]
)
def update_graph8(_):
    # Calculate fees per visit for urban and rural areas
    village_name = session.get('village_name', None)
    data_from_mongodb1 = list(collection1.find({"village_name": village_name}))
    urban_fees_per_visit = sum(int(data.get('urban_fees', '0')) / max(1, int(data.get('urban_visits', '1'))) for data in data_from_mongodb1)
    rural_fees_per_visit = sum(int(data.get('rural_fees', '0')) / max(1, int(data.get('rural_visits', '1'))) for data in data_from_mongodb1)

    return {
        'data': [
            go.Bar(
                x=['Urban Visits', 'Rural Visits'],
                y=[urban_fees_per_visit, rural_fees_per_visit],
                marker=dict(color=['#007bff', '#6f42c1'])
            )
        ],
        'layout': {
            'title': 'Fees per Visit (Urban vs. Rural)',
            'xaxis': {'title': 'Type of Visit'},
            'yaxis': {'title': 'Fees per Visit'}
        }
    }

@app_dash9.callback(
    dash.dependencies.Output('hospital-type-bar-chart', 'figure'),
    [dash.dependencies.Input('hospital-type-bar-chart', 'id')]
)
def update_graph8(_):
    # Fetch data from MongoDB collection for hospital type
    village_name = session.get('village_name', None)
    data_from_mongodb = list(collection1.find({"village_name": village_name}))
    
    # Count occurrences of each hospital type
    hospital_type_counts = {
        'government': 0,
        'private': 0,
        'both': 0
    }
    
    for data in data_from_mongodb:
        hospital_type = data.get('hospital_type', '').lower()
        if hospital_type in hospital_type_counts:
            hospital_type_counts[hospital_type] += 1

    # Prepare data for the bar chart
    x_values = list(hospital_type_counts.keys())
    y_values = list(hospital_type_counts.values())

    return {
        'data': [
            go.Bar(
                x=x_values,
                y=y_values,
                marker=dict(color=['#ff9999', '#66b3ff', '#99ff99'])  # Adjust colors if needed
            )
        ],
        'layout': {
            'title': 'Distribution of Hospital Types',
            'xaxis': {'title': 'Hospital Type'},
            'yaxis': {'title': 'Count'}
        }
    }














# Define layout for Dash app9
app_dash10.layout = html.Div(children=[
    dcc.Graph(
        id='income-food-dependency-scatter-plot'
    )
])

@app_dash10.callback(
    dash.dependencies.Output('income-food-dependency-scatter-plot', 'figure'),
    [dash.dependencies.Input('income-food-dependency-scatter-plot', 'id')]
)
def update_graph9(_):
    # Fetch data from MongoDB collection for nutrition analysis
    village_name = session.get('village_name', None)
    data_from_mongodb = list(collection4.find({"village_name": village_name}))
    
    # Extract relevant data for income levels and food dependency
    income_levels = [int(data.get('income', 0)) for data in data_from_mongodb]
    food_dependency = [1 if data.get('income_dependent', '').lower() == 'yes' else 0 for data in data_from_mongodb]
    
    return {
        'data': [
            go.Scatter(
                x=income_levels,
                y=food_dependency,
                mode='markers',
                marker=dict(color='#ff9999')  # Adjust marker color if needed
            )
        ],
        'layout': {
            'title': 'Income vs. Food Dependency',
            'xaxis': {'title': 'Monthly Income'},
            'yaxis': {'title': 'Food Dependency (0: No, 1: Yes)'}
        }
    }


app_dash11.layout = html.Div(children=[
    dcc.Graph(
        id='fruits-vegetables-frequency-donut-chart'
    )
])

@app_dash11.callback(
    dash.dependencies.Output('fruits-vegetables-frequency-donut-chart', 'figure'),
    [dash.dependencies.Input('fruits-vegetables-frequency-donut-chart', 'id')]
)
def update_graph10(_):
    # Fetch data from MongoDB collection for nutrition analysis
    village_name = session.get('village_name', None)
    data_from_mongodb = list(collection4.find({"village_name": village_name}))
    
    # Extract relevant data for fruits and vegetables frequency
    fruits_frequency = [int(data.get('fruits_frequency', 0)) for data in data_from_mongodb]
    vegetables_frequency = [int(data.get('vegetables_frequency', 0)) for data in data_from_mongodb]
    
    # Count households that consume fruits more frequently than vegetables, vice versa, or equally
    fruits_greater_than_vegetables = sum(1 for f, v in zip(fruits_frequency, vegetables_frequency) if f > v)
    vegetables_greater_than_fruits = sum(1 for f, v in zip(fruits_frequency, vegetables_frequency) if f < v)
    equal_fruits_and_vegetables = sum(1 for f, v in zip(fruits_frequency, vegetables_frequency) if f == v)
    
    # Define labels and values for the donut chart
    labels = ['Fruits > Vegetables', 'Fruits < Vegetables', 'Fruits = Vegetables']
    values = [fruits_greater_than_vegetables, vegetables_greater_than_fruits, equal_fruits_and_vegetables]
    
    return {
        'data': [
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99'])  # Adjust colors if needed
            )
        ],
        'layout': {
            'title': 'Comparison of Fruits and Vegetables Consumption Frequency'
        }
    }



@app_dash12.callback(
    dash.dependencies.Output('primary-food-source-affordability-horizontal-bar-chart', 'figure'),
    [dash.dependencies.Input('primary-food-source-affordability-horizontal-bar-chart', 'id')]
)
def update_graph11(_):
    # Fetch data from MongoDB collection for nutrition analysis
    village_name = session.get('village_name', None)
    data_from_mongodb = list(collection4.find({"village_name": village_name}))
    
    # Extract relevant data for primary food source and affordability of fruits
    primary_food_source = [data.get('primary_food_source', '') for data in data_from_mongodb]
    affordability_of_fruits = [data.get('afford_fruits', '') for data in data_from_mongodb]
    
    # Calculate the proportion of households that can afford fruits for each primary food source
    categories = set(primary_food_source)
    afford_fruits_proportion = {}
    
    for category in categories:
        total_households = primary_food_source.count(category)
        affordable_count = sum(1 for src, aff in zip(primary_food_source, affordability_of_fruits) if src == category and aff == 'yes')
        afford_fruits_proportion[category] = affordable_count / total_households if total_households > 0 else 0
    
 
    sorted_categories = sorted(categories, key=lambda x: afford_fruits_proportion[x], reverse=True)
    
   
    x_values = [afford_fruits_proportion[category] for category in sorted_categories]
    y_values = sorted_categories
    
    return {
        'data': [
            go.Bar(
                x=x_values,
                y=y_values,
                orientation='h',
                marker=dict(color='#660099'),  
                opacity=0.7
            )
        ],
        'layout': {
            'title': 'Proportion of Households That Can Afford Fruits by Primary Food Source',
            'xaxis': {'title': 'Proportion of Households'},
            'yaxis': {'title': 'Primary Food Source'},
            'margin': {'l': 150}  
        }
    }


@app_dash13.callback(
    dash.dependencies.Output('household-size-daily-food-consumption-bar-chart', 'figure'),
    [dash.dependencies.Input('household-size-daily-food-consumption-bar-chart', 'id')]
)
def update_graph12(_):
    # Fetch data from MongoDB collection for nutrition analysis
    village_name = session.get('village_name', None)
    data_from_mongodb = list(collection4.find({"village_name": village_name}))
    
    # Extract relevant data for household size and daily food consumption
    household_sizes = [data.get('household_size', 0) for data in data_from_mongodb]
    daily_food_consumption = [data.get('food_consumption', 'no') for data in data_from_mongodb]
    
    # Calculate the count of households that consume food daily for each household size
    household_size_counts = {}
    
    for size, consumption in zip(household_sizes, daily_food_consumption):
        if consumption == 'yes':
            household_size_counts[size] = household_size_counts.get(size, 0) + 1
    
    # Prepare data for the bar chart
    x_values = list(household_size_counts.keys())
    y_values = list(household_size_counts.values())
    
    return {
        'data': [
            go.Bar(
                x=x_values,
                y=y_values,
                marker=dict(color='#6633FF'),  # Adjust color if needed
                opacity=0.7
            )
        ],
        'layout': {
            'title': 'Household Size and Daily Food Consumption',
            'xaxis': {'title': 'Household Size'},
            'yaxis': {'title': 'Number of Households'},
            'plot_bgcolor': 'rgb(240, 240, 240)',  # Set plot background color
            'paper_bgcolor': 'rgb(255, 255, 255)',  # Set paper background color
            'font': {'color': 'rgb(50, 50, 50)'}  # Set font color
        }
    }



@app_dash14.callback(
    dash.dependencies.Output('clean-water-water-source-donut-chart', 'figure'),
    [dash.dependencies.Input('clean-water-water-source-donut-chart', 'id')]
)
def update_graph14(_):
    # Fetch data from MongoDB collection
    village_name = session.get('village_name', None)
    data_from_mongodb = list(collection4.find({"village_name": village_name}))
    
    # Count occurrences of clean water and water source
    clean_water_counts = sum(1 for data in data_from_mongodb if data.get('clean_water') == 'yes')
    not_clean_water_counts = len(data_from_mongodb) - clean_water_counts
    
    water_source_counts = {
        'common': sum(1 for data in data_from_mongodb if data.get('water_source') == 'common'),
        'mineral': sum(1 for data in data_from_mongodb if data.get('water_source') == 'mineral')
    }
    
    # Prepare data for the donut chart
    labels = ['Clean Water', 'Not Clean Water', 'Common Water Source', 'Mineral Water Source']
    values = [clean_water_counts, not_clean_water_counts, water_source_counts['common'], water_source_counts['mineral']]
    
    return {
        'data': [
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(colors=['#1bcfb4', '#ff9999', '#66b3ff', '#99ff99'])
            )
        ],
        'layout': {
            'title': 'Clean Water and Water Source Distribution'
        }
    }



@app_dash15.callback(
    dash.dependencies.Output('income-children-milk-scatter-plot', 'figure'),
    [dash.dependencies.Input('income-children-milk-scatter-plot', 'id')]
)
def update_graph15(_):
    # Fetch data from MongoDB collection
    village_name = session.get('village_name', None)
    data_from_mongodb = list(collection4.find({"village_name": village_name}))
    
    # Extract relevant data for income and children's milk consumption
    income_levels = [int(data.get('income', 0)) for data in data_from_mongodb]
    children_milk = [1 if data.get('children_milk', '') == 'yes' else 0 for data in data_from_mongodb]
    
    # Create a scatter plot
    figure = {
        'data': [
            go.Scatter(
                x=income_levels,
                y=children_milk,
                mode='markers',
                marker=dict(color='#ff9999')  # Adjust marker color if needed
            )
        ],
        'layout': {
            'title': 'Income vs. Children\'s Milk Consumption',
            'xaxis': {'title': 'Income'},
            'yaxis': {'title': 'Children\'s Milk Consumption (0: No, 1: Yes)'}
        }
    }
    
    return figure

@app_dash16.callback(
    dash.dependencies.Output('income-children-milk-scatter-plot', 'figure'),
    [dash.dependencies.Input('income-children-milk-scatter-plot', 'id')]
)
def update_graph16(_):
    # Fetch data from MongoDB collection for pregnant women analysis
    data_from_mongodb = list(collection5.find({}))
    
    # Extract relevant data for Vitamin Intake and Nutrient Intake
    vitamin_intake = [data.get('vitamin_intake', 'no') for data in data_from_mongodb]
    nutrient_intake = [data.get('nutrient_intake', 'no') for data in data_from_mongodb]
    
    # Calculate the proportion of women with sufficient nutrient intake based on vitamin intake
    categories = set(vitamin_intake)
    sufficient_nutrient_intake_count = {}
    
    for category in categories:
        total_women = vitamin_intake.count(category)
        sufficient_count = sum(1 for vit, nut in zip(vitamin_intake, nutrient_intake) if vit == category and nut == 'yes')
        sufficient_nutrient_intake_count[category] = sufficient_count / total_women if total_women > 0 else 0
    
    # Prepare data for the bar chart
    x_values = list(categories)
    y_values = [sufficient_nutrient_intake_count[category] for category in categories]
    
    return {
        'data': [
            go.Bar(
                x=x_values,
                y=y_values,
                marker=dict(color='#000099'),  # Adjust color if needed
                opacity=0.7
            )
        ],
        'layout': {
            'title': 'Proportion of Women with Sufficient Nutrient Intake by Vitamin Intake',
            'xaxis': {'title': 'Vitamin Intake'},
            'yaxis': {'title': 'Proportion of Women with Sufficient Nutrient Intake'}
        }
    }


# Define callback to update graph dynamically
@app_dash17.callback(
    dash.dependencies.Output('financial-barriers-government-schemes-donut-chart', 'figure'),
    [dash.dependencies.Input('financial-barriers-government-schemes-donut-chart', 'id')]
)
def update_donut_chart(_):
    # Fetch data from MongoDB collection for pregnant women analysis
    village_name = session.get('village_name', None)
    data_from_mongodb = list(collection5.find({"village_name": village_name}))
    
    # Extract relevant data for Financial Barriers and Government Schemes
    financial_barriers = [data.get('financial_barriers', '') for data in data_from_mongodb]
    government_schemes = [data.get('government_schemes', '') for data in data_from_mongodb]
    
    # Count households facing financial barriers
    with_financial_barriers = sum(1 for barrier in financial_barriers if barrier == 'cost_of_services')
    
    # Count households benefitting from government schemes
    with_government_schemes = sum(1 for scheme in government_schemes if scheme == 'yes')
    
    # Calculate the proportion of households with financial barriers and government schemes
    total_households = len(data_from_mongodb)
    without_financial_barriers = total_households - with_financial_barriers
    without_government_schemes = total_households - with_government_schemes
    
    # Define labels and values for the donut chart
    labels = ['Financial Barriers', 'Government Schemes']
    values = [with_financial_barriers, with_government_schemes]
    
    return {
        'data': [
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(colors=['#33FF99 ', '#FF3366'])  # Adjust colors if needed
            )
        ],
        'layout': {
            'title': 'Financial Barriers and Government Schemes',
            'annotations': [
                {
                    'font': {'size': 15},
                    'showarrow': False,
                    'text': f"Total Households: {total_households}",
                    'x': 0.5,
                    'y': 0.5
                }
            ]
        }
    }


@app_dash18.callback(
    dash.dependencies.Output('delivery-preference-donut-chart', 'figure'),
    [dash.dependencies.Input('delivery-preference-donut-chart', 'id')]
)
def update_graph18(_):
    # Fetch data from MongoDB collection for pregnancy analysis
    village_name = session.get('village_name', None)
    data_from_mongodb = list(collection5.find({"village_name": village_name}))
    
    # Extract delivery preference data
    delivery_preference = [data.get('delivery_preference', '') for data in data_from_mongodb]
    
    # Count households with private and government delivery preference
    private_count = delivery_preference.count('private')
    government_count = delivery_preference.count('government')
    
    # Define labels and values for the donut chart
    # for this
    labels = ['Private', 'Government']
    values = [private_count, government_count]
    
    return {
        'data': [
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(colors=['#ff9999', '#66b3ff'])  
            )
        ],
        'layout': {
            'title': 'Delivery Preference',
            'margin': {'l': 20, 'r': 20, 't': 50, 'b': 20}
        }
    }

@app_dash19.callback(
    dash.dependencies.Output('healthcare-distance-box-plot', 'figure'),
    [dash.dependencies.Input('healthcare-distance-box-plot', 'id')]
)
def update_box_plot(_):
    # Fetch healthcare distance data from MongoDB
    healthcare_distances = [record['healthcare_distance'] for record in collection5.find({}, {'healthcare_distance': 1})]

    # Create the box plot trace
    trace = go.Box(y=healthcare_distances, name='Healthcare Distance')

    # Define layout for the box plot
    layout = go.Layout(
        title='Distribution of Healthcare Distances',
        yaxis=dict(title='Healthcare Distance')
    )

    # Return figure containing the box plot
    return {'data': [trace], 'layout': layout}






# Define route to render the dashboard with village name parameter
@app_flask.route('/dashboard/<village_name>')
def dashboard(village_name):
    global accessibility_values, num_members_vaccinated_values
    village_data = list(collection.find({"village_name": village_name}))
    village=village_name
    district_result = collection2.find_one({'village': village}, {'_id': 0, 'district': 1})
    district = district_result['district'] if district_result else None
   
    total_family_members_list = [int(doc.get('total_family_members', 0)) for doc in village_data]
    total_vaccinated_polio = sum(doc.get('polio_vaccinated', 0) for doc in village_data)
    total_not_vaccinated_polio = len(village_data) - total_vaccinated_polio
    total_covid_vaccinated = sum(doc.get('covid_vaccinated', 0) for doc in village_data)
    total_family_members = sum(total_family_members_list)
#transportation data
    village_data1 = list(collection1.find({"village_name": village_name}))
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

   
    total_healthcare_spending = sum(int(doc.get('healthcare_spending', 0)) for doc in village_data1)
    total_families = len(village_data1)
    average_healthcare_spending_per_family = total_healthcare_spending / total_families
    government_hospitals_count = sum(1 for doc in village_data1 if doc.get('hospital_type') == 'government')
    private_hospitals_count = sum(1 for doc in village_data1 if doc.get('hospital_type') == 'private')
    both_hospitals_count = sum(1 for doc in village_data1 if doc.get('hospital_type') == 'both')

    urban_fees_per_visit = sum(int(data.get('urban_fees', '0')) / max(1, int(data.get('urban_visits', '1'))) for data in village_data1)
    rural_fees_per_visit = sum(int(data.get('rural_fees', '0')) / max(1, int(data.get('rural_visits', '1'))) for data in village_data1)

    data = list(collection4.find({'village_name': village_name}))
    df = pd.DataFrame(data)

    # Statistical summaries for numeric fields
    numeric_fields = ['income', 'fruits_frequency', 'vegetables_frequency', 'household_size']
    numeric_summary = df[numeric_fields].describe()

    # Counts for categorical fields
    categorical_fields = ['income_dependent', 'primary_food_source', 'food_consumption', 'afford_fruits', 'clean_water', 'water_source', 'children_milk', 'children_nutrition']
    categorical_counts = {field: df[field].value_counts() for field in categorical_fields}

    # Calculate proportions for categorical fields
    categorical_proportions = {field: df[field].value_counts(normalize=True) for field in categorical_fields}
    data = collection5.find({'village_name': village_name}, {'prenatal_care': 1, 'visit_frequency': 1, 'complications': 1,
                                'healthcare_distance': 1, 'transportation': 1, 'financial_barriers': 1,
                                'doctor_fees': 1, 'government_schemes': 1, 'meals_per_day': 1})

    

    no_prenatal_care = 0
    complications = 0
    limited_healthcare_access = 0
    financial_barriers = 0
    high_doctor_fees = 0
    underutilized_government_schemes = 0
    nutritional_concerns = 0

    for entry in data:
        if entry['prenatal_care'] == 'no':
            no_prenatal_care += 1
        if entry['complications'] == 'yes':
            complications += 1
        if int(entry['healthcare_distance']) > 30 or entry['transportation'] == 'limited':
            limited_healthcare_access += 1
        if entry['financial_barriers'] != 'no':
            financial_barriers += 1
        if int(entry['doctor_fees']) > 1000:
            high_doctor_fees += 1
        if entry['government_schemes'] == 'no':
            underutilized_government_schemes += 1
        if int(entry['meals_per_day']) < 3:
            nutritional_concerns += 1

    total_pregnant_women = collection.count_documents({})
    received_prenatal_care = total_pregnant_women - no_prenatal_care
    no_complications = total_pregnant_women - complications
    adequate_healthcare_access = total_pregnant_women - limited_healthcare_access
    no_financial_barriers = total_pregnant_women - financial_barriers
    low_doctor_fees = total_pregnant_women - high_doctor_fees
    utilized_government_schemes = total_pregnant_women - underutilized_government_schemes
    adequate_nutrition = total_pregnant_women - nutritional_concerns




    # Pass the fetched data to the template
    return render_template('dashboard.html', village_name=village_name,district=district,total_family_members=total_family_members,total_vaccinated_polio=total_vaccinated_polio,  total_not_vaccinated_polio=total_not_vaccinated_polio,
                           total_covid_vaccinated=total_covid_vaccinated,average_healthcare_spending_per_family=average_healthcare_spending_per_family,
                            government_hospitals_count=government_hospitals_count,
                           private_hospitals_count=private_hospitals_count,
                           both_hospitals_count=both_hospitals_count,
                           urban_fees_per_visit=urban_fees_per_visit,
                           rural_fees_per_visit=rural_fees_per_visit, categorical_counts=categorical_counts, categorical_proportions=categorical_proportions,current_datetime=current_datetime,
                           received_prenatal_care=received_prenatal_care,no_complications=no_complications,adequate_healthcare_access=adequate_healthcare_access,
                           no_financial_barriers=no_financial_barriers,low_doctor_fees=low_doctor_fees,utilized_government_schemes=utilized_government_schemes,
                          adequate_nutrition=adequate_nutrition,no_prenatal_care=no_prenatal_care, complications=complications,
                           limited_healthcare_access=limited_healthcare_access, financial_barriers=financial_barriers,
                           high_doctor_fees=high_doctor_fees, underutilized_government_schemes=underutilized_government_schemes,
                           nutritional_concerns=nutritional_concerns)


@app_flask.route('/export_csv', methods=['POST'])
def export_csv():
    village_name = request.form['village_name']
    authenticated_village_name = session.get('village_name')
    if village_name==authenticated_village_name:
        data_collection1 = list(collection.find({'village_name': village_name}))
        data_collection2 = list(collection4.find({'village_name': village_name}))
        data_collection3 = list(collection5.find({'village_name': village_name}))
        data_collection4 = list(collection1.find({'village_name': village_name}))
    
    # Add more collections as needed
# for exporting dataset or csv file 
    # Convert MongoDB data to DataFrames
    
        df_collection1 = pd.DataFrame(data_collection1)
        df_collection2 = pd.DataFrame(data_collection2)
        df_collection3 = pd.DataFrame(data_collection3)
        df_collection4 = pd.DataFrame(data_collection4)



    # Convert data from other collections to DataFrames as needed

    # Prepare CSV data for each collection
        csv_data_collection1 = BytesIO()
        df_collection1.to_csv(csv_data_collection1, index=False, encoding='utf-8')
        csv_data_collection1.seek(0)

        csv_data_collection2 = BytesIO()
        df_collection2.to_csv(csv_data_collection2, index=False, encoding='utf-8')
        csv_data_collection2.seek(0)

        csv_data_collection3 = BytesIO()
        df_collection3.to_csv(csv_data_collection2, index=False, encoding='utf-8')
        csv_data_collection3.seek(0)

        csv_data_collection4 = BytesIO()
        df_collection4.to_csv(csv_data_collection2, index=False, encoding='utf-8')
        csv_data_collection4.seek(0)
    # Prepare CSV data for other collections as needed

    # Create response with multiple CSV attachments
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
          zip_file.writestr(f"{village_name}_collection1_survey_data.csv", csv_data_collection1.getvalue())
          zip_file.writestr(f"{village_name}_collection2_survey_data.csv", csv_data_collection2.getvalue())
          zip_file.writestr(f"{village_name}_collection3_survey_data.csv", csv_data_collection3.getvalue())
          zip_file.writestr(f"{village_name}_collection4_survey_data.csv", csv_data_collection4.getvalue())
        # Add more CSV files to the zip archive as needed

        zip_buffer.seek(0)
    
        return send_file(zip_buffer,
                     as_attachment=True,
                     download_name=f"{village_name}_survey_data.zip",
                     mimetype='application/zip')
    else:
            # If the requested village name does not match the authenticated village name,
        # return an error response or redirect to an error page.
        return "Unauthorized Access: You are not authorized to download data for this village."

# Define index route

    
@app_flask.route('/dashboarddis/<district>')
def dashboarddis(district):
    # Query villages under the specified district
    villages_in_district = list(collection2.find({"district": district}))
    district_name = district
    total_family_members = 0
    total_vaccinated_polio = 0
    total_not_vaccinated_polio = 0
    total_covid_vaccinated = 0
    total_healthcare_spending = 0
    total_families = 0
    government_hospitals_count = 0
    private_hospitals_count = 0
    both_hospitals_count = 0
    urban_fees_per_visit = 0
    rural_fees_per_visit = 0
    
    doctor_fees_data = list(collection.find({}, {'doctor_fees': 1}))
    total_doctor_fees=0
    total_records=0
    average_doctor_fees = 0

    government_schemes_counts = {'yes': 0, 'no': 0}
    total_villages=0



    for village_data in villages_in_district:
        # Aggregate data from collection 2 (village vaccination data)
        village_vaccination_data = list(collection.find({"village_name": village_data['village']}))
        print(village_vaccination_data)
        total_family_members += sum(int(doc.get('total_family_members', 0)) for doc in village_vaccination_data)
        total_vaccinated_polio += sum(doc.get('polio_vaccinated', 0) for doc in village_vaccination_data)
        total_not_vaccinated_polio += max(0, len(village_vaccination_data) - total_vaccinated_polio)
        total_covid_vaccinated += sum(doc.get('covid_vaccinated', 0) for doc in village_vaccination_data)

        # Aggregate data from collection 1 (village healthcare data)
        village_healthcare_data = list(collection1.find({"village_name": village_data['village']}))
        total_healthcare_spending += sum(int(doc.get('healthcare_spending', 0)) for doc in village_healthcare_data)
        total_families += len(village_healthcare_data)
        government_hospitals_count += sum(1 for doc in village_healthcare_data if doc.get('hospital_type') == 'government')
        private_hospitals_count += sum(1 for doc in village_healthcare_data if doc.get('hospital_type') == 'private')
        both_hospitals_count += sum(1 for doc in village_healthcare_data if doc.get('hospital_type') == 'both')
        urban_fees_per_visit += sum(int(data.get('urban_fees', '0')) / max(1, int(data.get('urban_visits', '1'))) for data in village_healthcare_data)
        rural_fees_per_visit += sum(int(data.get('rural_fees', '0')) / max(1, int(data.get('rural_visits', '1'))) for data in village_healthcare_data)  

        village_pregencey_data = list(collection5.find({"village_name": village_data['village']}))
        total_doctor_fees +=sum(record.get('doctor_fees', 0) for record in doctor_fees_data)
        total_records += len(doctor_fees_data)
        average_doctor_fees += total_doctor_fees / total_records if total_records > 0 else 0
        data_from_mongodb = list(collection5.find({"village_name": village_pregencey_data}, {"government_schemes": 1}))

        for record in data_from_mongodb:
            scheme = record.get('government_schemes', 'no')
            government_schemes_counts[scheme] += 1

        total_villages += len(villages_in_district)


    

    # Pass the aggregated data to the template
    return render_template('disdash.html', district_name=district_name, total_family_members=total_family_members,
                           total_vaccinated_polio=total_vaccinated_polio, total_not_vaccinated_polio=total_not_vaccinated_polio,
                           total_covid_vaccinated=total_covid_vaccinated,
                           government_hospitals_count=government_hospitals_count,
                           private_hospitals_count=private_hospitals_count,
                           both_hospitals_count=both_hospitals_count,
                           urban_fees_per_visit=urban_fees_per_visit,
                           rural_fees_per_visit=rural_fees_per_visit
                           ,average_doctor_fees=average_doctor_fees,government_schemes_counts=government_schemes_counts,total_villages=total_villages)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Vaccination analysis >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
app_dash20 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard20/')

app_dash20.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='donut-chart')
])

@app_dash20.callback(
    dash.dependencies.Output('donut-chart', 'figure'),
    [dash.dependencies.Input('update-graph-button', 'n_clicks')],
    [dash.dependencies.State('donut-chart', 'id')]
)
def update_graph20(n_clicks, _):
    if n_clicks is None:
        return dash.no_update

    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))
    accessibility_values_temp = []
    num_members_vaccinated_values_temp = []

    for village_data in villages_in_district:
        village_data_access = list(collection.find({"village_name": village_data['village']}))
        village_data_vaccinated = list(collection1.find({"village_name": village_data['village']}))
        accessibility_values_temp.extend([doc['children_vaccinated'] for doc in village_data_access])
        num_members_vaccinated_values_temp.extend([doc.get('num_members_vaccinated', 0) for doc in village_data_vaccinated])

    return {
        'data': [
            go.Pie(
                labels=['Children Vaccinated', 'Number of Vaccinated Members'],
                values=[sum(accessibility_values_temp), sum(num_members_vaccinated_values_temp)],
                hole=0.4,
                marker=dict(colors=['#1bcfb4', '#a05aff'])
            )
        ],
        'layout': {
            'title': 'Summary: Accessibility vs Number of Vaccinated Members'
        }
    }

app_dash21 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard21/')

app_dash21.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='donut-chart2')
]) 

@app_dash21.callback(
    dash.dependencies.Output('donut-chart2', 'figure'),
    [dash.dependencies.Input('donut-chart2', 'id')]
)
def update_graph21(_):
    # Extract relevant data for visualization in app_dash2
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))
    accessibility_values=[]
    total_accessible=0

    for village_data in villages_in_district:
        village_data_access = list(collection.find({"village_name": village_data['village']}))
        accessibility_values.extend([doc['accessibility'] for doc in village_data_access])
        total_accessible = sum(accessibility_values)

    return {
        'data': [
            go.Pie(
                labels=['Accessibility Available', 'Accessibility Not Available'],
                values=[total_accessible, len(village_data_access) - total_accessible],
                hole=0.4,
                marker=dict(colors=['#007bff', '#6f42c1'])
            )
        ],
        'layout': {
            'title': 'Relationship between Community Engagement and Accessibility'
        }
    }

app_dash22 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard22/')

app_dash22.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='bar-graph')
]) 

@app_dash22.callback(
    dash.dependencies.Output('bar-graph', 'figure'),
    [dash.dependencies.Input('bar-graph', 'id')]
)
def update_graph22(_):
    # Fetch data from the MongoDB collection for Asha Worker Help
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))
    asha_worker_help_values=[]
    asha_worker_help_yes=0
    asha_worker_help_no=0
    for village_data in villages_in_district:
        village_data_access = list(collection.find({"village_name": village_data['village']}))
        asha_worker_help_values.extend([doc.get('asha_worker_help', 0) for doc in village_data_access])
        asha_worker_help_yes = sum(1 for val in asha_worker_help_values if val == 1)
        asha_worker_help_no = sum(1 for val in asha_worker_help_values if val == 0)
    

    return {
        'data': [
            go.Bar(
                x=['Yes', 'No'],
                y=[asha_worker_help_yes, asha_worker_help_no],
                marker=dict(color=['#8E44AD', '#00cccc'])
            )
        ],
        'layout': {
            'title': 'Relationship between Asha Worker Help and Total Family Members',
            'xaxis': {'title': 'Asha Worker Help'},
            'yaxis': {'title': 'Count'}
        }
    }

app_dash23 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard23/')

app_dash23.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='line-graph')
]) 

@app_dash23.callback(
    dash.dependencies.Output('line-graph', 'figure'),
    [dash.dependencies.Input('line-graph', 'id')]
)
def update_graph23(_):
    # Fetch data from the MongoDB collection for Awareness and Members Vaccinated
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))
    awareness_values=[]
    num_members_vaccinated_values=[]
    for village_data in villages_in_district:
        village_data_access = list(collection.find({"village_name": village_data['village']}))
        awareness_values.extend([doc.get('awareness', 0) for doc in village_data_access])
        num_members_vaccinated_values.extend([doc.get('num_members_vaccinated', 0) for doc in village_data_access])

    # Set line and marker colors
    line_color = '#1f77b4'  # Blue
    marker_color = '#ff7f0e'  # Orange

    return {
        'data': [
            go.Scatter(
                x=awareness_values,
                y=num_members_vaccinated_values,
                mode='lines+markers',
                line=dict(color=line_color),  # Set line color
                marker=dict(color=marker_color)  # Set marker color
            )
        ],
        'layout': {
            'title': 'Number of Members Vaccinated Over Time',
            'xaxis': {'title': 'Awareness (0: No, 1: Yes)'},
            'yaxis': {'title': 'Number of Members Vaccinated'}
        }
    }

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Vacccination End here>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>transportation analysis>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
app_dash24 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard24/')

app_dash24.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='income-transportation-spending')
]) 

@app_dash24.callback(
    dash.dependencies.Output('income-transportation-spending', 'figure'),
    [dash.dependencies.Input('income-transportation-spending', 'id')]
)
def update_graph24(_):
    # Fetch data from MongoDB collection for Income vs. Transportation Spending
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))
    
    # Aggregate data for Income and Transportation Spending
    income_values = []
    transportation_spending_values = []
    for village_data in villages_in_district:
        village_data_access = list(collection1.find({"village_name": village_data['village']}))
        income_values.extend([doc.get('income', 0) for doc in village_data_access])
        transportation_spending_values.extend([doc.get('transportation_spending', 0) for doc in village_data_access])

    # Set line and marker colors
    line_color = '#1f77b4'  # Blue
    marker_color = '#ff7f0e'  # Orange

    # Create the figure
    figure = {
        'data': [
            go.Scatter(
                x=income_values,
                y=transportation_spending_values,
                mode='markers',
                marker=dict(color=marker_color)  # Adjust marker color if needed
            )
        ],
        'layout': {
            'title': 'Income vs. Transportation Spending',
            'xaxis': {'title': 'Income'},
            'yaxis': {'title': 'Transportation Spending'}
        }
    }
    
    return figure

app_dash25 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard25/')

app_dash25.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='urban-rural-visits-bar-chart')
]) 


@app_dash25.callback(
    dash.dependencies.Output('urban-rural-visits-bar-chart', 'figure'),
    [dash.dependencies.Input('urban-rural-visits-bar-chart', 'id')]
)
def update_graph25(_):
    # Fetch data from MongoDB collection for Urban vs. Rural Visits
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))
    
    # Aggregate data for Urban and Rural Visits
    urban_visits_total = 0
    rural_visits_total = 0
    for village_data in villages_in_district:
        village_data_access = list(collection1.find({"village_name": village_data['village']}))
        urban_visits_total += sum(int(data.get('urban_visits', 0)) for data in village_data_access)
        rural_visits_total += sum(int(data.get('rural_visits', 0)) for data in village_data_access)

    # Create the figure
    figure = {
        'data': [
            go.Bar(
                x=['Urban Visits', 'Rural Visits'],
                y=[urban_visits_total, rural_visits_total],
                marker=dict(color=['#03A9F4', '#66FFCC'])
            )
        ],
        'layout': {
            'title': 'Urban vs. Rural Visits',
            'xaxis': {'title': 'Type of Visit'},
            'yaxis': {'title': 'Number of Visits'}
        }
    }
    
    return figure


app_dash26 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard26/')

app_dash26.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='fees-per-visit')
]) 

@app_dash26.callback(
    dash.dependencies.Output('fees-per-visit', 'figure'),
    [dash.dependencies.Input('fees-per-visit', 'id')]
)
def update_graph26(_):
    # Fetch data from MongoDB collection for Fees per Visit
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))

    # Initialize variables to calculate fees per visit for urban and rural areas
    urban_fees_total = 0
    rural_fees_total = 0
    urban_visits_total = 0
    rural_visits_total = 0

    # Aggregate data for urban and rural fees and visits
    for village_data in villages_in_district:
        village_data_analysis = list(collection1.find({"village_name": village_data['village']}))
        urban_fees_total += sum(int(data.get('urban_fees', 0)) for data in village_data_analysis)
        rural_fees_total += sum(int(data.get('rural_fees', 0)) for data in village_data_analysis)
        urban_visits_total += sum(int(data.get('urban_visits', 0)) for data in village_data_analysis)
        rural_visits_total += sum(int(data.get('rural_visits', 0)) for data in village_data_analysis)

    # Calculate fees per visit for urban and rural areas
    urban_fees_per_visit = urban_fees_total / max(1, urban_visits_total)
    rural_fees_per_visit = rural_fees_total / max(1, rural_visits_total)

    return {
        'data': [
            go.Bar(
                x=['Urban Visits', 'Rural Visits'],
                y=[urban_fees_per_visit, rural_fees_per_visit],
                marker=dict(color=['#007bff', '#6f42c1'])
            )
        ],
        'layout': {
            'title': 'Fees per Visit (Urban vs. Rural)',
            'xaxis': {'title': 'Type of Visit'},
            'yaxis': {'title': 'Fees per Visit'}
        }
    }


app_dash27 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard27/')

app_dash27.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='hospital-type-bar-chart')
]) 

@app_dash27.callback(
    dash.dependencies.Output('hospital-type-bar-chart', 'figure'),
    [dash.dependencies.Input('hospital-type-bar-chart', 'id')]
)
def update_graph27(_):
    # Fetch data from MongoDB collection for hospital type
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))

    # Initialize variables to count occurrences of each hospital type
    government_count = 0
    private_count = 0
    both_count = 0

    # Count occurrences of each hospital type for all villages in the district
    for village_data in villages_in_district:
        village_data_analysis = list(collection1.find({"village_name": village_data['village']}))
        for data in village_data_analysis:
            hospital_type = data.get('hospital_type', '').lower()
            if hospital_type == 'government':
                government_count += 1
            elif hospital_type == 'private':
                private_count += 1
            elif hospital_type == 'both':
                both_count += 1

    # Prepare data for the bar chart
    x_values = ['Government', 'Private', 'Both']
    y_values = [government_count, private_count, both_count]

    return {
        'data': [
            go.Bar(
                x=x_values,
                y=y_values,
                marker=dict(color=['#ff9999', '#66b3ff', '#99ff99'])  # Adjust colors if needed
            )
        ],
        'layout': {
            'title': 'Distribution of Hospital Types',
            'xaxis': {'title': 'Hospital Type'},
            'yaxis': {'title': 'Count'}
        }
    }

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>tranportation End here>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Nutarition Analysis start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
app_dash28 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard28/')

app_dash28.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='income-food-dependency-scatter-plot')
]) 

@app_dash28.callback(
    dash.dependencies.Output('income-food-dependency-scatter-plot', 'figure'),
    [dash.dependencies.Input('income-food-dependency-scatter-plot', 'id')]
)
def update_graph28(_):
    # Fetch data from MongoDB collection for nutrition analysis
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))

    # Initialize lists to store income levels and food dependency values
    income_levels = []
    food_dependency = []

    # Extract relevant data for income levels and food dependency for all villages in the district
    for village_data in villages_in_district:
        village_data_analysis = list(collection4.find({"village_name": village_data['village']}))
        for data in village_data_analysis:
            income_levels.append(int(data.get('income', 0)))
            food_dependency.append(1 if data.get('income_dependent', '').lower() == 'yes' else 0)

    return {
        'data': [
            go.Scatter(
                x=income_levels,
                y=food_dependency,
                mode='markers',
                marker=dict(color='#ff9999')  # Adjust marker color if needed
            )
        ],
        'layout': {
            'title': 'Income vs. Food Dependency',
            'xaxis': {'title': 'Monthly Income'},
            'yaxis': {'title': 'Food Dependency (0: No, 1: Yes)'}
        }
    }


app_dash29 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard29/')

app_dash29.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='fruits-vegetables-frequency-donut-chart')
]) 

@app_dash29.callback(
    dash.dependencies.Output('fruits-vegetables-frequency-donut-chart', 'figure'),
    [dash.dependencies.Input('fruits-vegetables-frequency-donut-chart', 'id')]
)
def update_graph29(_):
    # Fetch data from MongoDB collection for nutrition analysis
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))

    # Initialize counters for fruits and vegetables frequency comparison
    fruits_greater_than_vegetables = 0
    vegetables_greater_than_fruits = 0
    equal_fruits_and_vegetables = 0

    # Extract relevant data for fruits and vegetables frequency for all villages in the district
    for village_data in villages_in_district:
        village_data_analysis = list(collection4.find({"village_name": village_data['village']}))
        for data in village_data_analysis:
            fruits_frequency = int(data.get('fruits_frequency', 0))
            vegetables_frequency = int(data.get('vegetables_frequency', 0))
            if fruits_frequency > vegetables_frequency:
                fruits_greater_than_vegetables += 1
            elif fruits_frequency < vegetables_frequency:
                vegetables_greater_than_fruits += 1
            else:
                equal_fruits_and_vegetables += 1

    # Define labels and values for the donut chart
    labels = ['Fruits > Vegetables', 'Fruits < Vegetables', 'Fruits = Vegetables']
    values = [fruits_greater_than_vegetables, vegetables_greater_than_fruits, equal_fruits_and_vegetables]

    return {
        'data': [
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(colors=['#ff9999', '#66b3ff', '#99ff99'])  # Adjust colors if needed
            )
        ],
        'layout': {
            'title': 'Comparison of Fruits and Vegetables Consumption Frequency'
        }
    }


app_dash30 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard30/')

app_dash30.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='primary-food-source-affordability-horizontal-bar-chart')
]) 
@app_dash30.callback(
    dash.dependencies.Output('primary-food-source-affordability-horizontal-bar-chart', 'figure'),
    [dash.dependencies.Input('primary-food-source-affordability-horizontal-bar-chart', 'id')]
)
def update_graph30(_):
    # Fetch data from MongoDB collection for nutrition analysis
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))

    # Initialize dictionary to store the proportion of households that can afford fruits for each primary food source
    afford_fruits_proportion = {}

    # Extract relevant data for primary food source and affordability of fruits for all villages in the district
    for village_data in villages_in_district:
        village_data_analysis = list(collection4.find({"village_name": village_data['village']}))
        for data in village_data_analysis:
            primary_food_source = data.get('primary_food_source', '')
            affordability_of_fruits = data.get('afford_fruits', '')
            if primary_food_source not in afford_fruits_proportion:
                afford_fruits_proportion[primary_food_source] = {'total': 0, 'affordable': 0}
            afford_fruits_proportion[primary_food_source]['total'] += 1
            if affordability_of_fruits == 'yes':
                afford_fruits_proportion[primary_food_source]['affordable'] += 1

    # Calculate the proportion of households that can afford fruits for each primary food source
    for key, value in afford_fruits_proportion.items():
        if value['total'] > 0:
            value['proportion'] = value['affordable'] / value['total']
        else:
            value['proportion'] = 0

    # Sort categories by their afford fruits proportion
    sorted_categories = sorted(afford_fruits_proportion.keys(), key=lambda x: afford_fruits_proportion[x]['proportion'], reverse=True)

    # Prepare data for the horizontal bar chart
    x_values = [afford_fruits_proportion[category]['proportion'] for category in sorted_categories]
    y_values = sorted_categories

    return {
        'data': [
            go.Bar(
                x=x_values,
                y=y_values,
                orientation='h',
                marker=dict(color='#660099'),  # Adjust color if needed
                opacity=0.7
            )
        ],
        'layout': {
            'title': 'Proportion of Households That Can Afford Fruits by Primary Food Source',
            'xaxis': {'title': 'Proportion of Households'},
            'yaxis': {'title': 'Primary Food Source'},
            'margin': {'l': 150}  # Adjust left margin for longer category labels
        }
    }

app_dash31 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard31/')

app_dash31.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='household-size-daily-food-consumption-bar-chart')
]) 
@app_dash31.callback(
    dash.dependencies.Output('household-size-daily-food-consumption-bar-chart', 'figure'),
    [dash.dependencies.Input('household-size-daily-food-consumption-bar-chart', 'id')]
)
def update_graph31(_):
    # Fetch data from MongoDB collection for nutrition analysis
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))

    # Initialize dictionary to store the count of households that consume food daily for each household size
    household_size_counts = {}

    # Extract relevant data for household size and daily food consumption for all villages in the district
    for village_data in villages_in_district:
        village_data_analysis = list(collection4.find({"village_name": village_data['village']}))
        for data in village_data_analysis:
            household_size = data.get('household_size', 0)
            daily_food_consumption = data.get('food_consumption', 'no')
            if daily_food_consumption == 'yes':
                household_size_counts[household_size] = household_size_counts.get(household_size, 0) + 1

    # Prepare data for the bar chart
    x_values = list(household_size_counts.keys())
    y_values = list(household_size_counts.values())

    return {
        'data': [
            go.Bar(
                x=x_values,
                y=y_values,
                marker=dict(color='#6633FF'),  # Adjust color if needed
                opacity=0.7
            )
        ],
        'layout': {
            'title': 'Household Size and Daily Food Consumption',
            'xaxis': {'title': 'Household Size'},
            'yaxis': {'title': 'Number of Households'},
            'plot_bgcolor': 'rgb(240, 240, 240)',  # Set plot background color
            'paper_bgcolor': 'rgb(255, 255, 255)',  # Set paper background color
            'font': {'color': 'rgb(50, 50, 50)'}  # Set font color
        }
    }

app_dash32 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard32/')

app_dash32.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='clean-water-water-source-donut-chart')
]) 
@app_dash32.callback(
    dash.dependencies.Output('clean-water-water-source-donut-chart', 'figure'),
    [dash.dependencies.Input('clean-water-water-source-donut-chart', 'id')]
)
def update_graph32(_):
    # Fetch data from MongoDB collection
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))

    # Initialize counters for clean water and water source
    clean_water_counts = 0
    not_clean_water_counts = 0
    common_water_counts = 0
    mineral_water_counts = 0

    # Count occurrences of clean water and water source for all villages in the district
    for village_data in villages_in_district:
        village_data_analysis = list(collection4.find({"village_name": village_data['village']}))
        for data in village_data_analysis:
            if data.get('clean_water') == 'yes':
                clean_water_counts += 1
            else:
                not_clean_water_counts += 1
            if data.get('water_source') == 'common':
                common_water_counts += 1
            elif data.get('water_source') == 'mineral':
                mineral_water_counts += 1

    # Prepare data for the donut chart
    labels = ['Clean Water', 'Not Clean Water', 'Common Water Source', 'Mineral Water Source']
    values = [clean_water_counts, not_clean_water_counts, common_water_counts, mineral_water_counts]

    return {
        'data': [
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(colors=['#1bcfb4', '#ff9999', '#66b3ff', '#99ff99'])
            )
        ],
        'layout': {
            'title': 'Clean Water and Water Source Distribution'
        }
    }

app_dash33 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard33/')

app_dash33.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='income-children-milk-scatter-plot')
]) 

@app_dash33.callback(
    dash.dependencies.Output('income-children-milk-scatter-plot', 'figure'),
    [dash.dependencies.Input('income-children-milk-scatter-plot', 'id')]
)
def update_graph33(_):
    # Fetch data from MongoDB collection
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))
    
    # Initialize lists to store aggregated data
    income_levels = []
    children_milk = []
    
    # Extract relevant data for income and children's milk consumption from all villages in the district
    for village_data in villages_in_district:
        data_from_mongodb = list(collection4.find({"village_name": village_data['village']}))
        for data in data_from_mongodb:
            income_levels.append(int(data.get('income', 0)))
            children_milk.append(1 if data.get('children_milk', '') == 'yes' else 0)
    
    # Create a scatter plot
    figure = {
        'data': [
            go.Scatter(
                x=income_levels,
                y=children_milk,
                mode='markers',
                marker=dict(color='#ff9999')  # Adjust marker color if needed
            )
        ],
        'layout': {
            'title': 'Income vs. Children\'s Milk Consumption in {}'.format(district),
            'xaxis': {'title': 'Income'},
            'yaxis': {'title': 'Children\'s Milk Consumption (0: No, 1: Yes)'}
        }
    }
    
    return figure

app_dash34 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard34/')

app_dash34.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='nutrient-intake-bar-chart')
]) 

@app_dash34.callback(
    dash.dependencies.Output('nutrient-intake-bar-chart', 'figure'),
    [dash.dependencies.Input('nutrient-intake-bar-chart', 'id')]
)
def update_graph34(_):
    # Fetch data from MongoDB collection for pregnant women analysis
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))
    
    # Initialize dictionary to store the proportion of women with sufficient nutrient intake
    sufficient_nutrient_intake_count = {'yes': 0, 'no': 0}
    
    # Extract relevant data for vitamin intake and nutrient intake from all villages in the district
    for village_data in villages_in_district:
        data_from_mongodb = list(collection5.find({"village_name": village_data['village']}))
        for data in data_from_mongodb:
            vitamin_intake = data.get('vitamin_intake', 'no')
            nutrient_intake = data.get('nutrient_intake', 'no')
            if nutrient_intake == 'yes':
                sufficient_nutrient_intake_count[vitamin_intake] = sufficient_nutrient_intake_count.get(vitamin_intake, 0) + 1
    
    # Calculate the proportion of women with sufficient nutrient intake based on vitamin intake
    total_women = sum(sufficient_nutrient_intake_count.values())
    proportion_yes = sufficient_nutrient_intake_count['yes'] / total_women if total_women > 0 else 0
    proportion_no = sufficient_nutrient_intake_count['no'] / total_women if total_women > 0 else 0
    
    # Prepare data for the donut chart
    labels = ['Vitamin Intake Yes', 'Vitamin Intake No']
    values = [proportion_yes, proportion_no]
    
    return {
        'data': [
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(colors=['#1bcfb4', '#ff9999'])  # Adjust colors if needed
            )
        ],
        'layout': {
            'title': 'Proportion of Women with Sufficient Nutrient Intake by Vitamin Intake in {}'.format(district)
        }
    }




# >>>>>>>>>>>>>>>>>>>>>>>>>>>>nutration End Here>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# <>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>pregencey analysis >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

app_dash35 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard35/')

app_dash35.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='financial-barriers-government-schemes-donut-chart')
]) 

@app_dash35.callback(
    dash.dependencies.Output('financial-barriers-government-schemes-donut-chart', 'figure'),
    [dash.dependencies.Input('financial-barriers-government-schemes-donut-chart', 'id')]
)
def update_donut_chart(_):
    # Fetch data from MongoDB collection for pregnant women analysis
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))
    
    # Initialize counts for financial barriers and government schemes
    with_financial_barriers = 0
    with_government_schemes = 0
    
    # Iterate through all villages in the district
    for village_data in villages_in_district:
        data_from_mongodb = list(collection5.find({"village_name": village_data['village']}))
        for data in data_from_mongodb:
            financial_barriers = data.get('financial_barriers', '')
            government_schemes = data.get('government_schemes', '')
            if financial_barriers == 'cost_of_services':
                with_financial_barriers += 1
            if government_schemes == 'yes':
                with_government_schemes += 1
    
    # Calculate households without financial barriers and government schemes
    total_households = len(villages_in_district)
    without_financial_barriers = total_households - with_financial_barriers
    without_government_schemes = total_households - with_government_schemes
    
    # Define labels and values for the donut chart
    labels = ['Financial Barriers', 'Government Schemes']
    values = [with_financial_barriers, with_government_schemes]
    
    return {
        'data': [
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(colors=['#33FF99', '#FF3366'])  # Adjust colors if needed
            )
        ],
        'layout': {
            'title': 'Financial Barriers and Government Schemes in {}'.format(district),
            'annotations': [
                {
                    'font': {'size': 15},
                    'showarrow': False,
                    'text': f"Total Households: {total_households}",
                    'x': 0.5,
                    'y': 0.5
                }
            ]
        }
    }


app_dash36 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard36/')

app_dash36.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='delivery-preference-donut-chart')
]) 
@app_dash36.callback(
    dash.dependencies.Output('delivery-preference-donut-chart', 'figure'),
    [dash.dependencies.Input('delivery-preference-donut-chart', 'id')]
)
def update_graph36(_):
    # Fetch data from MongoDB collection for pregnancy analysis
    district = session.get('district', None)
    villages_in_district = list(collection2.find({"district": district}))
    
    # Initialize variables to count delivery preferences
    private_count_total = 0
    government_count_total = 0
    
    # Iterate through each village in the district
    for village_data in villages_in_district:
        # Fetch data from MongoDB collection for the current village
        data_from_mongodb = list(collection5.find({"village_name": village_data['village']}))
        
        # Extract delivery preference data for the current village
        delivery_preference = [data.get('delivery_preference', '') for data in data_from_mongodb]
        
        # Count households with private and government delivery preference for the current village
        private_count = delivery_preference.count('private')
        government_count = delivery_preference.count('government')
        
        # Accumulate counts for all villages in the district
        private_count_total += private_count
        government_count_total += government_count
    
    # Define labels and values for the donut chart
    labels = ['Private', 'Government']
    values = [private_count_total, government_count_total]
    
    return {
        'data': [
            go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                marker=dict(colors=['#ff9999', '#66b3ff'])  # Adjust colors if needed
            )
        ],
        'layout': {
            'title': 'Delivery Preference',
            'margin': {'l': 20, 'r': 20, 't': 50, 'b': 20}
        }
    }

app_dash37 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard37/')

app_dash37.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='healthcare-distance-box-plot')
]) 
@app_dash37.callback(
    dash.dependencies.Output('healthcare-distance-box-plot', 'figure'),
    [dash.dependencies.Input('healthcare-distance-box-plot', 'id')]
)
def update_box_plot(_):
    # Get the district from the session
    district = session.get('district', None)
   

    # Fetch healthcare distance data from MongoDB for all villages in the district
    healthcare_distances = []
    for village_data in collection2.find({"district": district}, {"village": 1}):
        village_name = village_data['village']
        data_from_mongodb = list(collection5.find({"village_name": village_name}, {"healthcare_distance": 1}))
        healthcare_distances.extend([record['healthcare_distance'] for record in data_from_mongodb])

    # Create the box plot trace
    trace = go.Box(y=healthcare_distances, name='Healthcare Distance')

    # Define layout for the box plot
    layout = go.Layout(
        title='Distribution of Healthcare Distances',
        yaxis=dict(title='Healthcare Distance')
    )

    # Return figure containing the box plot
    return {'data': [trace], 'layout': layout}


app_dash38 = dash.Dash(__name__, server=app_flask, url_base_pathname='/dashboard38/')

app_dash38.layout = html.Div(children=[
    html.Button('Update Graph', id='update-graph-button'),
    dcc.Graph(id='government-schemes-pie-chart')
]) 
@app_dash38.callback(
    dash.dependencies.Output('government-schemes-pie-chart', 'figure'),
    [dash.dependencies.Input('government-schemes-pie-chart', 'id')]
)
def update_pie_chart(_):
    # Get the district from the session
    district = session.get('district', None)
    
    # Fetch data from MongoDB collection for all villages in the district
    government_schemes_counts = {'yes': 0, 'no': 0}
    for village_data in collection2.find({"district": district}, {"village": 1}):
        village_name = village_data['village']
        data_from_mongodb = list(collection5.find({"village_name": village_name}, {"government_schemes": 1}))
        for record in data_from_mongodb:
            scheme = record.get('government_schemes', 'no')
            government_schemes_counts[scheme] += 1

    # Define labels and values for the pie chart
    labels = list(government_schemes_counts.keys())
    values = list(government_schemes_counts.values())

    # Create the pie chart trace
    trace = go.Pie(labels=labels, values=values, hole=0.4, marker=dict(colors=['#1bcfb4', '#FF3366  ']))

    # Define layout for the pie chart
    layout = go.Layout(title='Government Schemes Distribution')

    # Return figure containing the pie chart
    return {'data': [trace], 'layout': layout}

@app_flask.route('/ashaworkers')
def ashaworkers():
    return render_template('asha_worker_login.html')

# Define login route
@app_flask.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        village_name = request.form['village_name']
        session['village_name'] = village_name
        return redirect(url_for('dashboard', village_name=village_name))
    else:
        return 'Invalid Request'
    
@app_flask.route('/health1')
def health1():
    return render_template('healthworker.html')
    

@app_flask.route('/Health')
def Health():
   
     return render_template('ALL.html')


    # Define login route
@app_flask.route('/HW', methods=['POST'])
def HW():
    if request.method == 'POST':
        village_name = request.form['village_name']
        session['village_name'] = village_name
        return redirect(url_for('Health'))
    else:
        return 'Invalid Request'
    

@app_flask.route('/login1', methods=['POST'])
def login1():
    if request.method == 'POST':
        district = session.get('district', None)
        village_name = request.form['village_name']
        session['village_name'] = village_name
        village_count = collection2.count_documents({'village_name': village_name, 'district': district})
        if village_count>=0:
            return redirect(url_for('dashboard', village_name=village_name))
        else:
            return 'Invalid Request'
    

@app_flask.route('/index1')
def index1():
    return render_template('dislog.html')

# Define login route
@app_flask.route('/dislog', methods=['POST'])
def dislog():
    if request.method == 'POST':
        district = request.form['district']
        session['district'] = district
        return redirect(url_for('dashboarddis', district=district))
    else:
        return 'Invalid Request'

@app_flask.route('/')
def home():
    return render_template('Home_page.html')

@app_flask.route('/asha_worker')
def asha_worker():
    return render_template('asha_worker_register.html')



@app_flask.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        asha_group = request.form['asha-group']
        village = request.form['village']
        taluka = request.form['taluka']
        district = request.form['district']
        pin_code = request.form['pin-code']
        password = request.form['password']

        # Insert data into MongoDB
        data = {
            'asha_group': asha_group,
            'village': village,
            'taluka': taluka,
            'district': district,
            'zip_code': pin_code,
            'password': password
        }
        collection2.insert_one(data)
        districts = collection2.distinct('district')
        # Redirect to login page after successful registration
        return redirect(url_for('login'))

@app_flask.route('/submit_vaccination', methods=['POST'])
def submit_vaccination():
    # Get form data
    village_name = request.form['village_name']
    awareness = int(request.form['awareness'])
    children_vaccinated = int(request.form['children_vaccinated'])
    accessibility = int(request.form['accessibility'])
    trust = int(request.form['trust'])
    community_engagement = int(request.form['community_engagement'])
    num_members_vaccinated = int(request.form['num_members_vaccinated'])
    date = request.form['date'] 
    
    # Handle radio button inputs for vaccination status
    covid_vaccinated = int(request.form.get('covid_vaccinated'))
    polio_vaccinated = int(request.form.get('polio_vaccinated'))
    other_disease_vaccinated = int(request.form.get('other_disease_vaccinated'))

    # Get additional fields
    total_family_members = int(request.form['total_family_members'])
    asha_worker_help = int(request.form['asha_worker_help'])
    num_children = int(request.form['num_children'])
    num_adults = int(request.form['num_adults'])
    age_children = [int(age.strip()) for age in request.form['age_children'].split(',') if age.strip()]
    age_adults = [int(age.strip()) for age in request.form['age_adults'].split(',') if age.strip()]

    # Insert data into MongoDB
    data = {
        'village_name': village_name,
        'awareness': awareness,
        'children_vaccinated': children_vaccinated,
        'accessibility': accessibility,
        'trust': trust,
        'community_engagement': community_engagement,
        'num_members_vaccinated': num_members_vaccinated,
        'covid_vaccinated': covid_vaccinated,
        'polio_vaccinated': polio_vaccinated,
        'other_disease_vaccinated': other_disease_vaccinated,
        'total_family_members': total_family_members,
        'asha_worker_help': asha_worker_help,
        'num_children': num_children,
        'num_adults': num_adults,
        'age_children': age_children,
        'age_adults': age_adults,
         'date': date
    }
    collection.insert_one(data)

    return redirect(url_for('Health'))

@app_flask.route('/submit_transportation', methods=['POST'])
def submit_transportation():
    if request.method == 'POST':
        village_name = request.form['village_name']
        income = request.form['income']
        earning_members = request.form['earning_members']
        healthcare_spending = request.form['healthcare_spending']
        hospital_type = request.form['hospital_type']
        transportation_spending = request.form['transportation_spending']
        urban_visits = request.form['urban_visits']
        rural_visits = request.form['rural_visits']
        urban_fees = request.form['urban_fees']
        rural_fees = request.form['rural_fees']
        date = request.form['date']  # Get date from the form

        # Insert data into MongoDB
        survey_data = {
            'village_name': village_name,
            'income': income,
            'earning_members': earning_members,
            'healthcare_spending': healthcare_spending,
            'hospital_type': hospital_type,
            'transportation_spending': transportation_spending,
            'urban_visits': urban_visits,
            'rural_visits': rural_visits,
            'urban_fees': urban_fees,
            'rural_fees': rural_fees,
            'date': date  # Store the date in the database
        }
        collection1.insert_one(survey_data)

        return redirect(url_for('Health'))

@app_flask.route('/submit_nutrition', methods=['POST'])
def submit_nutrition():
    if request.method == 'POST':
        # Extract data from the form
        village_name = request.form['village_name']
        income = int(request.form['income'])
        income_dependent = request.form['income_dependent']
        fruits_frequency = int(request.form['fruits_frequency'])
        vegetables_frequency = int(request.form['vegetables_frequency'])
        primary_food_source = request.form['primary_food_source']
        household_size = int(request.form['household_size'])
        food_consumption = request.form['food_consumption']
        afford_fruits = request.form['afford_fruits']
        clean_water = request.form['clean_water']
        water_source = request.form['water_source']
        children_milk = request.form['children_milk']
        children_nutrition = request.form['children_nutrition']
        today_date = request.form['date']

        # Prepare document to insert into MongoDB
        nutrition_data = {
            'village_name': village_name,
            'income': income,
            'income_dependent': income_dependent,
            'fruits_frequency': fruits_frequency,
            'vegetables_frequency': vegetables_frequency,
            'primary_food_source': primary_food_source,
            'household_size': household_size,
            'food_consumption': food_consumption,
            'afford_fruits': afford_fruits,
            'clean_water': clean_water,
            'water_source': water_source,
            'children_milk': children_milk,
            'children_nutrition': children_nutrition,
            'date': today_date
        }

        # Insert document into MongoDB
        collection4.insert_one(nutrition_data)

        return redirect(url_for('Health'))

@app_flask.route('/submit_pregnant', methods=['POST'])
def submit_pregnant():
    if request.method == 'POST':
        # Extract data from the form
        village_name = request.form['village_name']
        prenatal_care = request.form['prenatal_care']
        visit_frequency = request.form['visit_frequency']
        complications = request.form['complications']
        vitamin_intake = request.form['vitamin_intake']
        meals_per_day = int(request.form['meals_per_day'])
        nutrient_intake = request.form['nutrient_intake']
        healthcare_distance = request.form['healthcare_distance']
        transportation = request.form['transportation']
        financial_barriers = request.form['financial_barriers']
        delivery_preference = request.form['delivery_preference']
        government_schemes = request.form['government_schemes']
        doctor_fees = float(request.form['doctor_fees'])
        today_date = request.form['date']

        # Prepare document to insert into MongoDB
        pregnant_data = {
            'village_name': village_name,
            'prenatal_care': prenatal_care,
            'visit_frequency': visit_frequency,
            'complications': complications,
            'vitamin_intake': vitamin_intake,
            'meals_per_day': meals_per_day,
            'nutrient_intake': nutrient_intake,
            'healthcare_distance': healthcare_distance,
            'transportation': transportation,
            'financial_barriers': financial_barriers,
            'delivery_preference': delivery_preference,
            'government_schemes': government_schemes,
            'doctor_fees': doctor_fees,
            'date': today_date
        }

        # Insert document into MongoDB
        collection5.insert_one(pregnant_data)

        return redirect(url_for('Health'))
    
@app_flask.route('/submit_benefits', methods=['POST'])
def submit_benefits():
    if request.method == 'POST':
        # Extract data from the form
        village_name = request.form['village_name']
        scheme_name = request.form['scheme_name']
        scheme_review = request.form['scheme_review']
        today_date = request.form['date']

        # Prepare document to insert into MongoDB
        benefits_data = {
            'village_name': village_name,
            'scheme_name': scheme_name,
            'scheme_review': scheme_review,
            'date': today_date
        }

        # Insert document into MongoDB
        collection7.insert_one(benefits_data)

        return redirect(url_for('Health'))
    

    
# db = client['hackfest']
# collection = db['vaccination']
# collection1 = db['transportation']
# collection2 = db['asha_worker_register']
# collection4=db['nutritionForm']
# collection5=db['Pwomen']
# collection7 = db['Gscheme']


# db = client['hackfest1']
# collection3 = db['vaccination']
# collection4=db['transportation']
# collection5= db['nutritionForm']
# collection6= db['Pwomen']
# collection7 = db['Gscheme']
# data_collection5 = list(collection5.find())

@app_flask.route('/generate_report', methods=['POST'])
def generate_report():
    village_name = request.form['village_name']
    report_path = generate_analysis_report(village_name)
    if report_path and os.path.exists(report_path):
    # Set the download_name parameter to provide the filename for the attachment
       return send_file(report_path, as_attachment=True, mimetype='application/pdf', download_name='analysis_report.pdf')
    else:
    # Handle the case where the PDF report is not generated or not found
       return "Error: PDF report not found or not generated."
    


def generate_analysis_report(village_name):
      
    # Read the dataset from CSV file
    data_collection1 = list(collection1.find({'village_name': village_name}))
    data_collection2 = list(collection4.find({'village_name': village_name}))
    data_collection3 = list(collection.find({'village_name': village_name}))
    df = pd.DataFrame(data_collection1)
    df1 = pd.DataFrame(data_collection2)
   
    df2 = pd.DataFrame(data_collection3)
    print(df1.head())


    # Define income categories
    def income_category(income):
        if income <= 30000:  # Low income
            return 'Low'
        elif income <= 100000:  # Medium income
            return 'Medium'
        else:  # High income
            return 'High'


    # Categorize incomes
    df1['income_group'] = df1['income'].apply(income_category)

    # Define nutritional values for fruits
    nutritional_values_fruits = {
        'Calories': 67.25,
        'Calories from Fat': 1.75,
        'Total Fat (g)': 0.175,
        'Sodium (mg)': 1.75,
        'Potassium (mg)': 30.25,
        'Total Carb. (g)': 17.75,
        'Dietary Fiber (g)': 2.6,
        'Sugars (g)': 9.15,
        'Protein (g)': 0.575,
        'Vitamin A (%DV)': 6.35,
        'Vitamin C (%DV)': 2.15,
        'Calcium (%DV)': 1.55,
        'Iron (%DV)': 0.55
    }

    # Multiply the average nutritional values by fruit consumption for each income group
    for nutrient, avg_value in nutritional_values_fruits.items():
        df1[nutrient] = df1['fruits_frequency'] * avg_value

    # Define nutritional values for vegetables
    nutritional_values_veg = {
        'Calories': 20.2,
        'Calories from Fat': 0.8,
        'Total Fat (g)': 0.08,
        'Sodium (mg)': 18.4,
        'Potassium (mg)': 72.8,
        'Total Carb. (g)': 4.5,
        'Dietary Fiber (g)': 1.6,
        'Sugars (g)': 2.8,
        'Protein (g)': 1.0,
        'Vitamin A (%DV)': 10.0,
        'Vitamin C (%DV)': 7.0,
        'Calcium (%DV)': 2.4,
        'Iron (%DV)': 2.2
    }

    # Multiply the average nutritional values by vegetable consumption for each income group
    for nutrient, avg_value in nutritional_values_veg.items():
        df1[nutrient] = df1['vegetables_frequency'] * avg_value

    # Calculate average frequency of fruits and vegetables consumed by each income group
    avg_freq_fruit = df1.groupby('income_group').agg(
        {'fruits_frequency': 'mean'})
    print(avg_freq_fruit)

    avg_freq_veg = df1.groupby('income_group').agg(
        {'vegetables_frequency': 'mean'})
    print(avg_freq_fruit)

    # Save images of nutritional consumption
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Plot nutritional consumption of fruits
    for nutrient, avg_value in nutritional_values_fruits.items():
        df1[nutrient] = df1['fruits_frequency'] * avg_value

    for group, data in df1.groupby('income_group'):
        data.set_index('income_group')[list(
            nutritional_values_fruits.keys())].mean().plot(ax=ax1, label=group)

    ax1.set_ylabel('Level of Nutrients')
    ax1.set_xlabel('Nutritional Categories')
    ax1.set_title('Nutritional Consumption of Fruits by Income Group')
    ax1.legend()

    # Save image for nutritional consumption of fruits
    fig1.savefig('nutritional_consumption_fruits.png')

    # Close the plot to release memory
    plt.close(fig1)

    # Save images of nutritional consumption
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Plot nutritional consumption of vegetables
    for nutrient, avg_value in nutritional_values_veg.items():
        df1[nutrient] = df1['vegetables_frequency'] * avg_value

    for group, data in df1.groupby('income_group'):
        data.set_index('income_group')[list(
            nutritional_values_veg.keys())].mean().plot(ax=ax2, label=group)

    ax2.set_ylabel('Level of Nutrients')
    ax2.set_xlabel('Nutritional Categories')
    ax2.set_title('Nutritional Consumption of Vegetables by Income Group')
    ax2.legend()

    # Save image for nutritional consumption of vegetables
    fig2.savefig('nutritional_consumption_vegetables.png')

    # Close the plot to release memory
    plt.close(fig2)

    nutritional_data = [
        ["Nutrient", "Low Income", "Medium Income", "High Income"],
    ]

    # Add nutritional values for fruits
    for nutrient, avg_value in nutritional_values_fruits.items():
        row = [nutrient]
        for group, freq in avg_freq_fruit.iterrows():
            row.append(round(freq['fruits_frequency'] * avg_value, 2))
        nutritional_data.append(row)

    # Add nutritional values for vegetables
    for nutrient, avg_value in nutritional_values_veg.items():
        row = [nutrient]
        for group, freq in avg_freq_veg.iterrows():
            row.append(round(freq['vegetables_frequency'] * avg_value,
                            2))
        nutritional_data.append(row)

    # Create table for nutritional consumption
    nutritional_table = Table(nutritional_data)

    # Calculate whether families need to consume more fruits and vegetables based on the minimum consumption standards
    fruits_consumption_status = {}
    for index, row in avg_freq_fruit.iterrows():
        if row['fruits_frequency'] < 4:
            fruits_consumption_status[index] = 'Less consumption'
        else:
            fruits_consumption_status[index] = 'Sufficient consumption'

    veg_consumption_status = {}
    for index, row in avg_freq_veg.iterrows():
        if row['vegetables_frequency'] < 6:
            veg_consumption_status[index] = 'Less consumption'
        else:
            veg_consumption_status[index] = 'Sufficient consumption'

    # Plot the consumption graph
    avg_freq = pd.concat([avg_freq_fruit, avg_freq_veg], axis=1)
    avg_freq.plot(kind='bar', figsize=(10, 6), color=['skyblue', 'lightgreen'])
    plt.title(
        'Average Frequency of Fruit and Vegetable Consumption by Income Group')
    plt.xlabel('Income Group')
    plt.ylabel('Average Frequency')
    plt.xticks(rotation=0)
    plt.legend(['Fruits', 'Vegetables'])
    plt.savefig('consumption_graph.png')
    plt.close()
    consumption_graph = Image('consumption_graph.png',
                                    width=400,
                                    height=300)

    conclusion = "\n<b>Conclusion:</b>\n"
    conclusion += "The average frequency of fruit consumption across income groups is as follows:\n"
    conclusion += avg_freq_fruit.to_string() + "\n\n"
    conclusion += "The average frequency of vegetable consumption across income groups is as follows:\n"
    conclusion += avg_freq_veg.to_string() + "\n\n"

    # Check if families are meeting minimum consumption standards
    conclusion += "Based on the minimum consumption standards:\n"
    for index, row in avg_freq.iterrows():
        if row['fruits_frequency'] >= 4 and row['vegetables_frequency'] >= 6:
            conclusion += f"Families in the {index} income group are consuming a sufficient amount of fruits and vegetables.\n"
        else:
            conclusion += f"Families in the {index} income group need to consume more fruits and/or vegetables to meet the minimum standards.\n"
    # Pie chart for clean water consumption
    clean_water_counts = df1['clean_water'].value_counts(normalize=True)
    plt.figure(figsize=(8, 6))
    plt.pie(clean_water_counts,
            labels=clean_water_counts.index,
            autopct='%1.1f%%',
            startangle=140)
    plt.title('Percentage of People Consuming Clean Water')
    plt.axis('equal')
    plt.savefig('clean_water_pie.png')
    plt.close()
    clean_water_pie= Image('clean_water_pie.png',
                                    width=400,
                                    height=300)
    clean_water_percentage = clean_water_counts.get('yes', 0) * 100

    # Derive conclusion for clean water consumption
    if clean_water_percentage >= 95:
        water_quality_conclusion = 'Water quality is good.'
    else:
        water_quality_conclusion = f'Water quality needs to improve. Only {clean_water_percentage:}% of people have access to clean water.'

    conclusion_text = f"Conclusion: Water quality needs to improve. Only {clean_water_percentage:.1f}% of people have access to clean water."

    # Set threshold for government alert
    alert_threshold = 95

    # Check if clean water percentage is lower than the alert threshold
    if clean_water_percentage < alert_threshold:
        government_alert = f"The percentage of people consuming clean water is below {alert_threshold}%. Government needs to take immediate action to improve water quality."
    else:
        government_alert = "The percentage of people consuming clean water is above the threshold. Water quality is considered safe."

        
    # Pie chart for water source
    water_source_counts = df1['water_source'].value_counts(normalize=True)
    plt.figure(figsize=(8, 6))
    plt.pie(water_source_counts,
            labels=water_source_counts.index,
            autopct='%1.1f%%',
            startangle=140)
    plt.title('Percentage of People Using Different Water Sources')
    plt.axis('equal')
    plt.savefig('water_source_pie.png')
    plt.close()
    water_source_pie= Image('water_source_pie.png',
                                    width=400,
                                    height=300)

    # Conclusion for water sources
    non_clean_water_sources = water_source_counts[water_source_counts.index.isin(
        df1[df1['clean_water'] == 'no']['water_source'])]
    if not non_clean_water_sources.empty:
        water_source_conclusion = "Water sources needing improvement:\n"
        for source in non_clean_water_sources.index:
            water_source_conclusion += f"More than 20% of people consume non-clean water from {source}. Government needs to take action to improve water quality.\n"
    else:
        water_source_conclusion = "No water source has more than 20% of people consuming non-clean water."

    # Pie chart for children drinking milk
    milk_counts = df1['children_milk'].value_counts(normalize=True)
    plt.figure(figsize=(8, 6))
    plt.pie(milk_counts,
            labels=milk_counts.index,
            autopct='%1.1f%%',
            startangle=140)
    plt.title('Percentage of Children Drinking Milk')
    plt.axis('equal')
    plt.savefig('children_milk_pie.png')
    plt.close()
    children_milk_pie= Image('children_milk_pie.png',
                                    width=400,
                                    height=300)

    percentage_milk = milk_counts.get('yes', 0) * 100
    if percentage_milk >= 95:
        milk_conclusion = f'{percentage_milk:.1f}% of children are getting milk. Enough children are getting milk.'
    else:
        milk_conclusion = f'Only {percentage_milk:.1f}% of children are getting milk. Government needs to take action to improve milk availability for children.'


    # Calculate percentage of income spent on healthcare
    df['healthcare_spending'] = pd.to_numeric(df['healthcare_spending'], errors='coerce')
    df['income'] = pd.to_numeric(df['income'], errors='coerce')
    df['healthcare_percentage'] = (df['healthcare_spending'] / df['income']) * 100

    # Calculate percentage of income spent on transportation
    df['transportation_spending'] = pd.to_numeric(df['transportation_spending'], errors='coerce')

# Now, perform the division
    df['transportation_percentage'] = (df['transportation_spending'] / df['income']) * 100

    # Analyze healthcare spending
    avg_healthcare_percentage = df['healthcare_percentage'].mean()
    healthcare_threshold = 10  # 10% of income
    if avg_healthcare_percentage > healthcare_threshold:
        healthcare_conclusion = "Healthcare spending is relatively high. Government intervention may be needed."
    else:
        healthcare_conclusion = "Healthcare spending is within reasonable limits."

    # Analyze transportation spending
    avg_transportation_percentage = df['transportation_percentage'].mean()
    transportation_threshold = 15  # 15% of income
    if avg_transportation_percentage > transportation_threshold:
        transportation_conclusion = "Transportation spending is relatively high. Government intervention may be needed."
    else:
        transportation_conclusion = "Transportation spending is within reasonable limits."

    # Hospital Type Distribution
    hospital_counts = df['hospital_type'].value_counts()
    total_entries = len(df)
    gov_percentage = (hospital_counts.get('government', 0) / total_entries) * 100
    private_percentage = (hospital_counts.get('private', 0) / total_entries) * 100
    both_percentage = (hospital_counts.get('both', 0) / total_entries) * 100

    # Set a threshold for government hospital percentage
    gov_hospital_threshold = 50  # 20% of total entries excluding 'both'

    # Conclusion for hospital type
    exclusive_total_entries = total_entries - hospital_counts.get('both', 0)
    exclusive_gov_percentage = (hospital_counts.get('government', 0) /
                                exclusive_total_entries) * 100

    if exclusive_gov_percentage >= gov_hospital_threshold:
        hospital_conclusion = "Percentage of people going to government hospitals is satisfactory."
    else:
        hospital_conclusion = "Percentage of people going to government hospitals is low. Government may need to improve the quality of government hospitals in the area."

    # Average Visits for Accessing Facilities
    avg_urban_visits = df['urban_visits'].mean()
    avg_rural_visits = df['rural_visits'].mean()

    # Check if rural areas have fewer resources compared to urban areas
    if avg_urban_visits > avg_rural_visits:
        visits_conclusion = "Rural areas have fewer resources compared to urban areas."
    else:
        visits_conclusion = "Rural areas have similar or more resources compared to urban areas."

    # Calculate total money spent on urban hospitals and rural hospitals
    df['urban_visits'] = pd.to_numeric(df['urban_visits'], errors='coerce')
    df['urban_fees'] = pd.to_numeric(df['urban_fees'], errors='coerce')
    df['rural_visits'] = pd.to_numeric(df['rural_visits'], errors='coerce')
    df['rural_fees'] = pd.to_numeric(df['rural_fees'], errors='coerce')

# Now, perform the multiplication
    df['urban_hospital_total'] = df['urban_visits'] * df['urban_fees'] 
    df['rural_hospital_total'] = df['rural_visits'] * df['rural_fees']

    urban_hospital_total = df['urban_hospital_total'].sum()
    rural_hospital_total = df['rural_hospital_total'].sum()

    # Calculate the difference and suggest increasing fees for rural hospitals
    average_difference = (urban_hospital_total -
                        rural_hospital_total) / total_entries
    suggestion = "Increase fees for rural hospitals to enhance medical resources in rural areas."

    fig1, ax1 = plt.subplots()
    ax1.pie([gov_percentage, private_percentage, both_percentage],
            labels=['Government', 'Private', 'Both'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['lightcoral', 'lightskyblue', 'lightgreen'])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Hospital Type Distribution')
    plt.savefig('hospital_type_distribution.png')  # Save the pie chart image
    plt.close()
    hospital_pie_chart = Image('hospital_type_distribution.png',
                            width=300,
                            height=300)

    plt.figure()
    df['healthcare_percentage'].plot(kind='hist', bins=5, color='lightblue')
    plt.title('Percentage of Income Spent on Healthcare')
    plt.xlabel('Percentage')
    plt.ylabel('Frequency')
    plt.savefig('healthcare_spending_distribution.png')  # Save the histogram image
    plt.close()
    healthcare_graph = Image('healthcare_spending.png', width=400, height=300)
    # Plot the Percentage of Income Spent on Transportation

    plt.figure()
    df['transportation_percentage'].plot(kind='hist', bins=5, color='lightgreen')
    plt.title('Percentage of Income Spent on Transportation')
    plt.xlabel('Percentage')
    plt.ylabel('Frequency')
    plt.savefig(
        'transportation_spending_distribution.png')  # Save the histogram image
    plt.close()
    transportation_graph = Image('transportation_spending.png',
                                width=400,
                                height=300)

    # Visualize Comparison of Urban and Rural Visits
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(['Urban', 'Rural'], [avg_urban_visits, avg_rural_visits],
            color=['blue', 'green'])
    plt.title('Average Visits for Accessing Facilities')
    plt.xlabel('Area')
    plt.ylabel('Average Visits')
    plt.savefig('visits_comparison.png')  # Save the bar chart image
    plt.close()
    visits_bar_chart = Image('visits_comparison.png', width=400, height=300)

    # Plot the Total Money Spend on Urban and Rural Hospitals
    plt.figure()
    plt.bar(['Urban', 'Rural'], [urban_hospital_total, rural_hospital_total],
            color=['blue', 'green'])
    plt.title('Total Money Spend on Urban vs. Rural Hospitals')
    plt.xlabel('Area')
    plt.ylabel('Total Money Spent ($)')
    plt.savefig('hospital_spending_comparison.png')  # Save the bar chart image
    plt.close()
    hospital_spending_graph = Image('hospital_spending_comparison.png',
                                    width=400,
                                    height=300)

    # Vaccination Report
    awareness_counts = df2['awareness'].value_counts(normalize=True)
    plt.figure(figsize=(8, 6))
    plt.pie(awareness_counts,
            labels=awareness_counts.index,
            autopct='%1.1f%%',
            startangle=140)
    plt.title('Percentage of People Aware of Vaccination Schemes')
    plt.axis('equal')
    plt.savefig('awareness_pie.png')
    plt.close()
    awareness_pie= Image('awareness_pie.png',
                                    width=400,
                                    height=300)

    # Conclusion for vaccination awareness
    awareness_threshold = 90
    if awareness_counts.get(1, 0) >= awareness_threshold:
        vaccination_conclusion = 'Over 90% of the population is aware of vaccination schemes.'
    else:
        vaccination_conclusion = f'Awareness about vaccination schemes is below {awareness_threshold}%. Government should focus on spreading more awareness.'

    # Calculate the total number of children and the total number of polio vaccinated children
    total_children = df2['children'].sum()
    total_polio_vaccinated = df2['polio_vaccinated'].sum()

    # Calculate the number of children who are not polio vaccinated
    not_polio_vaccinated = total_children - total_polio_vaccinated

    # Plot the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie([total_polio_vaccinated, not_polio_vaccinated],
            labels=['Polio Vaccinated', 'Not Polio Vaccinated'],
            autopct='%1.1f%%',
            startangle=140,
            colors=['lightgreen', 'lightcoral'])
    plt.title('Polio Vaccination Status among Children')
    plt.axis('equal')
    plt.savefig('polio_vaccination_pie.png')
    plt.close()
    polio_vaccination_pie= Image('polio_vaccination_pie.png',
                                    width=400,
                                    height=300)
    # Percentage of children polio vaccinated
    df2['polio_vaccination_percentage'] = (df2['polio_vaccinated'] /
                                        df2['children']) * 100
    polio_vaccination_percentage = df2['polio_vaccination_percentage'].mean()

    # Conclusion for polio vaccination percentage
    if polio_vaccination_percentage >= 90:
        polio_vaccination_conclusion = "Over 90% of children are polio vaccinated."
    else:
        polio_vaccination_conclusion = "Less than 90% of children are polio vaccinated. Government should vaccinate everyone."

    total_covid_vaccinated = df2['covid_vaccinated_children'].sum()

    # Calculate the number of children who are not COVID vaccinated
    not_covid_vaccinated = total_children - total_covid_vaccinated

    # Plot the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie([total_covid_vaccinated, not_covid_vaccinated],
            labels=['COVID Vaccinated', 'Not COVID Vaccinated'],
            autopct='%1.1f%%',
            startangle=140,
            colors=['lightblue', 'lightsalmon'])
    plt.title('COVID Vaccination Status among Children')
    plt.axis('equal')
    plt.savefig('covid_vaccination_pie.png')
    plt.close()
    covid_vaccination_pie= Image('covid_vaccination_pie.png',
                                    width=400,
                                    height=300)

    # Calculate the percentage of children who are COVID vaccinated
    percentage_covid_vaccinated = (total_covid_vaccinated / total_children) * 100

    # Derive conclusion from the analysis
    if percentage_covid_vaccinated >= 95:
        covid_vaccination_conclusion = 'Over 95% of children are COVID vaccinated.'
    else:
        covid_vaccination_conclusion = 'COVID vaccination coverage is below 95%. Government should focus on increasing vaccination rates.'

    print(covid_vaccination_conclusion)

    # Calculate the number of adults and their COVID vaccination status
    total_adults = df2['family_members'].sum() - df2['children'].sum()
    total_covid_vaccinated_adults = df2['covid_vaccinated_adults'].sum()
    not_covid_vaccinated_adults = total_adults - total_covid_vaccinated_adults

    # Ensure non-negative values for the pie chart
    total_covid_vaccinated_adults = max(0, total_covid_vaccinated_adults)
    not_covid_vaccinated_adults = max(0, not_covid_vaccinated_adults)

    # Create the pie chart for COVID vaccination among adults
    plt.figure(figsize=(8, 6))
    plt.pie([total_covid_vaccinated_adults, not_covid_vaccinated_adults],
            labels=['COVID Vaccinated', 'Not COVID Vaccinated'],
            autopct='%1.1f%%',
            startangle=140,
            colors=['lightblue', 'lightsalmon'])
    plt.title('COVID Vaccination Status among Adults')
    plt.axis('equal')
    plt.savefig('covid_vaccination_adults_pie.png')
    plt.close()
    covid_vaccination_adults_pie= Image('covid_vaccination_adults_pie.png',
                                    width=400,
                                    height=300)

    # Conclusion for COVID vaccination among adults
    covid_vaccination_adults_percentage = (total_covid_vaccinated_adults /
                                        total_adults) * 100
    if covid_vaccination_adults_percentage >= 95:
        covid_vaccination_adults_conclusion = "Over 95% of adults are COVID vaccinated."
    else:
        covid_vaccination_adults_conclusion = "Less than 95% of adults are COVID vaccinated. Government should focus on increasing adult vaccination coverage."

    satisfactory_service_count = df2['vaccination_service'].sum()
    total_villages = len(df2)
    unsatisfactory_service_count = total_villages - satisfactory_service_count

    # Create a bar chart for vaccination service quality
    plt.figure(figsize=(8, 6))
    plt.bar(['Satisfactory', 'Unsatisfactory'],
            [satisfactory_service_count, unsatisfactory_service_count],
            color=['lightgreen', 'lightcoral'])
    plt.title('Quality of Vaccination Service')
    plt.xlabel('Service Quality')
    plt.ylabel('Number of Villages')
    plt.savefig('vaccination_service_quality_bar.png')
    plt.close()
    vaccination_service_quality_bar= Image('vaccination_service_quality_bar.png',
                                    width=400,
                                    height=300)

    # Conclusion for vaccination service quality
    if satisfactory_service_count == total_villages:
        vaccination_service_conclusion = "Vaccination services are satisfactory for the entire village."
    else:
        vaccination_service_conclusion = "Vaccination services need improvement in some villages."


    # # Initialize MongoDB connection
    # # print("Connecting to MongoDB...")
    # # client = pymongo.MongoClient("mongodb://localhost:27017/")
    # # db = client["hackfest"]
    # # collection = db["feedback"]

    # # Initialize AWS services
    # print("Initializing AWS services...")
    # translate = boto3.client('translate', region_name=region_name)

    # # Initialize NLP pipeline with pre-trained model






    

    # Create PDF report
    doc = SimpleDocTemplate("analysis_report.pdf", pagesize=letter)
    
    # Sample styles
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph("Transportation Analysis Report", styles['Title'])
    title2 = Paragraph("Nutrition Analysis Report", styles['Title'])
    title3 = Paragraph("Vaccination Analysis Report", styles['Title'])
    

    # Text and Graphs in Tabular Format
    data = [
        [Paragraph("<b>Healthcare Spending Analysis:</b><br/>"
                    "Average Percentage of Income Spent on Healthcare: {:.2f}%<br/>"
                    "Conclusion: {}".format(avg_healthcare_percentage, healthcare_conclusion), styles['Normal']),
        Image('healthcare_spending_distribution.png', width=300, height=200)],
        [Paragraph("<b>Transportation Spending Analysis:</b><br/>"
                    "Average Percentage of Income Spent on Transportation: {:.2f}%<br/>"
                    "Conclusion: {}".format(avg_transportation_percentage, transportation_conclusion), styles['Normal']),
        Image('transportation_spending_distribution.png', width=300, height=200)],
        [Paragraph("<b>Hospital Type Distribution:</b><br/>"
                    "{}".format(hospital_conclusion), styles['Normal']),
        Image('hospital_type_distribution.png', width=300, height=200)],
        [Paragraph("<b>Average Visits for Accessing Facilities:</b><br/>"
                    "Urban: {:.2f}<br/>"
                    "Rural: {:.2f}<br/>"
                    "{}".format(avg_urban_visits, avg_rural_visits, visits_conclusion), styles['Normal']),
        Image('visits_comparison.png', width=300, height=200)],
        [Paragraph("<b>Difference and Suggestion:</b><br/>"
                    "Average Difference: ${:.2f}<br/>"
                    "Suggestion: {}".format(average_difference, suggestion), styles['Normal']),
        Image('hospital_spending_comparison.png', width=300, height=200)]
    ]

    data2 = [
        [
            Paragraph("<b>Nutritional Consumption of Fruits by Income Group:</b><br/>"),
            Image('nutritional_consumption_fruits.png', width=300, height=200)
        ],
        [
            Paragraph("<b>Nutritional Consumption of Vegetables by Income Group:</b><br/>"),
            Image('nutritional_consumption_vegetables.png', width=300, height=200)
        ],
        [
            Paragraph("<b>Food Consumption:</b><br/>"
                        "{}".format(conclusion), styles['Normal']),
            Image('consumption_graph.png', width=300, height=200)
        ],
        [
            Paragraph("<b>Clean Water Consumption:</b><br/>"
                    "{}".format(conclusion_text), styles['Normal']),
            Image('clean_water_pie.png', width=300, height=200)
        ],
        [
            Paragraph("<b>Water Source:</b><br/>"
                    "{}".format(water_source_conclusion), styles['Normal']),
            Image('water_source_pie.png', width=300, height=200)
        ],
        [
            Paragraph("<b>Percentage of Children Drinking Milk:</b><br/>"
                    "{}".format(milk_conclusion),styles['Normal']),
            Image('children_milk_pie.png', width=300, height=200)
        ]
    ]

    data3 = [
        [Paragraph("<b>Vaccination Awareness:</b><br/>"
                    "{}".format(vaccination_conclusion), styles['Normal']),
        Image('awareness_pie.png', width=300, height=200)],
        [Paragraph("<b>Relationship between Number of Children and Polio Vaccinated:</b><br/>"
                    "{}".format(polio_vaccination_conclusion), styles['Normal']),
        Image('polio_vaccination_pie.png', width=300, height=200)],
        [Paragraph("<b>COVID Vaccination Status among Children:</b><br/>"
                    "{}".format(covid_vaccination_conclusion), styles['Normal']),
        Image('covid_vaccination_pie.png', width=300, height=200)],
        [Paragraph("<b>COVID Vaccination Status among Adults:</b><br/>"
                    "{}".format(covid_vaccination_adults_conclusion), styles['Normal']),
        Image('covid_vaccination_adults_pie.png', width=300, height=200)],
        [Paragraph("<b>Vaccination Service Quality:</b><br/>"
                    "{}".format(vaccination_service_conclusion), styles['Normal']),
        Image('vaccination_service_quality_bar.png', width=300, height=200)]
    ]


    table = Table(data)
    table2 = Table(data2)
    table3 = Table(data3)
    table_width = 400
    # Add table style
    table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    table2.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    table3.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))


    nutritional_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    # Build the PDF report
    # Build the PDF report
    doc.build([title, table, PageBreak(), title2, table2, PageBreak(), title3, table3, PageBreak()])
    return "analysis_report.pdf"



# Run the Flask app
if __name__ == '__main__':
    app_flask.run(debug=True)
