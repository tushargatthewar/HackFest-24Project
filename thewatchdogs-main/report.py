import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph

styles = getSampleStyleSheet()
from reportlab.platypus import PageBreak
import os
from pymongo import MongoClient
import boto3
from safetensors import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline
from transformers.models import gpt2
import torch
import pandas as pd
from flask import Flask, render_template, request, send_file
import os

app = Flask(__name__)

client = MongoClient('mongodb://localhost:27017/')
db = client['hackfest1']
collection3 = db['vaccination']
collection4=db['transportation']
collection5= db['nutritionForm']
collection6= db['Pwomen']
collection7 = db['Gscheme']
data_collection5 = list(collection5.find())

df1 = pd.DataFrame(data_collection5)
print(df1.columns)




@app.route('/')
def index():
    return render_template('report.html')

@app.route('/generate_report', methods=['POST'])
def generate_report():
    village_name = request.form['village_name']
    report_path = generate_analysis_report(village_name)
    return send_file(report_path, as_attachment=True)

def download_and_load_gpt2_model():
        print("Checking GPT-2 model files...")
        # Check if the model files already exist
        if not os.path.exists("models/774M"):
            print("Downloading GPT-2 model...")
            # Download the GPT-2 model
            gpt2.download_gpt2(model_name="774M", model_dir="models")

        print("Loading GPT-2 model...")
        # Load the GPT-2 model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        return tokenizer, model

region_name = 'ap-south-1'


nlp = pipeline("sentiment-analysis")
comprehend = boto3.client('comprehend', region_name=region_name)
tokenizer, model = download_and_load_gpt2_model()
# Global variable for storing key issues and explanations
issues_explained = {}

def generate_analysis_report(village_name):
    # Read the dataset from CSV file
    data_collection1 = list(collection4.find({'village_name': village_name}))
    data_collection2 = list(collection5.find({'village_name': village_name}))
    data_collection3 = list(collection3.find({'village_name': village_name}))
    df = pd.DataFrame(data_collection1)
    df1 = pd.DataFrame(data_collection2)
   
    df2 = pd.DataFrame(data_collection3)
    print(df1)


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

    percentage_milk = milk_counts.get('yes', 0) * 100
    if percentage_milk >= 95:
        milk_conclusion = f'{percentage_milk:.1f}% of children are getting milk. Enough children are getting milk.'
    else:
        milk_conclusion = f'Only {percentage_milk:.1f}% of children are getting milk. Government needs to take action to improve milk availability for children.'


    # Calculate percentage of income spent on healthcare
    df['healthcare_percentage'] = (df['healthcare_spending'] / df['income']) * 100

    # Calculate percentage of income spent on transportation
    df['transportation_percentage'] = (df['transportation_spending'] /
                                    df['income']) * 100

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






    # Function to update the issues_explained dictionary based on feedbacks
    def update_issues_explained(feedbacks):
        global issues_explained  # Access the global dictionary

        for feedback in feedbacks:
            feedback_text = str(feedback['feedback']) if pd.notna(
                feedback['feedback']) else ""  # Convert to string or handle NaN
            sentiment = nlp(feedback_text)[0]['label']

            # Extract key phrases or entities only from negative feedbacks
            if sentiment == 'NEGATIVE':
                comprehend_response_entities = comprehend.detect_entities(Text=feedback_text, LanguageCode='en')
                entities = [entity['Text'] for entity in comprehend_response_entities['Entities']]

                comprehend_response_key_phrases = comprehend.detect_key_phrases(Text=feedback_text, LanguageCode='en')
                key_phrases = [phrase['Text'] for phrase in comprehend_response_key_phrases['KeyPhrases']]

                # Update the issues_explained dictionary with new key phrases or entities
                for issue in key_phrases + entities:
                    if issue not in issues_explained:
                        # Generate explanatory sentence using GPT-2
                        explanatory_sentence = generate_explanatory_sentence(issue)
                        issues_explained[issue] = explanatory_sentence


    # Generate explanatory sentence using GPT-2 with attention mask and pad token ID
    def generate_explanatory_sentence(issue):
        # Concatenate the issue with a prompt for GPT-2
        prompt = f"The issue '{issue}' with the scheme is: "
        input_text = prompt + issue

        # Tokenize the input text
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate the attention mask
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

        # Generate explanatory sentence using GPT-2 with attention mask
        with torch.no_grad():
            output = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id,
                                    max_length=50, num_return_sequences=1)

        # Decode the generated output
        explanatory_sentence = tokenizer.decode(output[0], skip_special_tokens=True)

        return explanatory_sentence


    # # Function to analyze aggregated feedbacks for a scheme
    def analyze_aggregated_feedbacks(feedbacks):
        # Combine all feedback texts for the same scheme into a single text
        all_feedbacks_text = ' '.join(str(feedback['feedback']) for feedback in feedbacks if pd.notna(feedback['feedback']))

        # Analyze sentiment of the combined feedback text
        sentiment = nlp(all_feedbacks_text)[0]['label']

        # Perform thematic content analysis for potential issues
        print_thematic_content(all_feedbacks_text, sentiment)

        # Summarize the identified problem for the scheme
        if sentiment == 'NEGATIVE':
            problem = "Negative sentiment detected."
        else:
            problem = "No significant issues were identified with the mentioned government scheme."

        # Debug prints

        return problem


    # Function to print thematic content for potential issues
    def print_thematic_content(all_feedbacks_text, sentiment):
        # Use AWS Comprehend to extract key phrases and entities
        comprehend_response_entities = comprehend.batch_detect_entities(TextList=[all_feedbacks_text], LanguageCode='en')
        entities = set(entity['Text'] for entity in comprehend_response_entities['ResultList'][0]['Entities'])

        comprehend_response_key_phrases = comprehend.batch_detect_key_phrases(TextList=[all_feedbacks_text],
                                                                            LanguageCode='en')
        key_phrases = set(phrase['Text'] for phrase in comprehend_response_key_phrases['ResultList'][0]['KeyPhrases'])



    # Function to summarize negatively polarized sentiment of feedbacks for a scheme
    def summarize_negative_sentiment(feedbacks):
        negative_phrases = set()

        # Update issues_explained dynamically
        update_issues_explained(feedbacks)

        for feedback in feedbacks:
            feedback_text = str(feedback['feedback']) if pd.notna(
                feedback['feedback']) else ""  # Convert to string or handle NaN
            sentiment = nlp(feedback_text)[0]['label']

            # If the sentiment is negative, extract key phrases or entities
            if sentiment == 'NEGATIVE':
                comprehend_response_entities = comprehend.detect_entities(Text=feedback_text, LanguageCode='en')
                entities = [entity['Text'] for entity in comprehend_response_entities['Entities']]

                comprehend_response_key_phrases = comprehend.detect_key_phrases(Text=feedback_text,
                                                                                LanguageCode='en')
                key_phrases = [phrase['Text'] for phrase in comprehend_response_key_phrases['KeyPhrases']]

                # Add negative key phrases and entities to the set
                negative_phrases.update(entities)
                negative_phrases.update(key_phrases)

        # Generate the summary paragraph
        if negative_phrases:
            issues_explained = {
                "work": "Many participants expressed dissatisfaction with the amount of work required for the scheme.",
                "Errands": "Participants mentioned facing numerous errands or tasks associated with the scheme.",
                "hobbies": "There were concerns raised about the scheme interfering with participants' hobbies or leisure activities.",
                "a lot": "Participants felt overwhelmed by the amount of work or tasks involved in the scheme.",
                "no awareness": "There was a lack of awareness or understanding about the scheme among the participants.",
                "any benefit": "Participants questioned whether there were any tangible benefits or advantages to the scheme.",
                "a job": "Some participants raised concerns about the scheme's impact on their employment or job prospects.",
                "able people": "Issues were raised regarding the scheme's accessibility for differently-abled individuals.",
                "weak scheme": "Criticism was directed towards the perceived weaknesses or shortcomings of the scheme.",
                "the world": "Participants expressed frustration about the scheme's inability to address broader issues or concerns.",
                "US": "Concerns were raised about the relevance or applicability of the scheme in the context of the United States.",
                "a serious problem": "Participants highlighted significant problems or issues associated with the scheme.",
                "this scheme": "There were criticisms directed towards specific aspects or elements of the scheme.",
                "Poor service": "Issues were raised regarding the quality or adequacy of the services provided by the scheme.",
                "a critical issue": "Participants identified critical issues or challenges that need urgent attention within the scheme."
            }

            # Generate the summary paragraph
            summarized_paragraph = "After analyzing the feedbacks for this scheme, it appears that there are several common negative sentiments expressed by the participants. "
            summarized_paragraph += "These sentiments include: "
            summarized_paragraph += ". ".join([issues_explained.get(issue, issue) for issue in negative_phrases]) + "."
            return summarized_paragraph

        else:
            summary = "After analyzing the feedbacks for this scheme, it appears that no key negative issues were identified."

        return summary


    # Function to process feedback for a given scheme
    def process_feedback_for_scheme(selected_scheme, feedbacks):

        # Print separator line

        # Analyze aggregated feedbacks for the selected scheme
        aggregated_problem = analyze_aggregated_feedbacks(feedbacks)

        # Ensure aggregated_problem is a string
        if not isinstance(aggregated_problem, str):
            aggregated_problem = str(aggregated_problem)

        # Print the identified problem for the selected scheme


        # Summarize negatively polarized sentiment

        summary = summarize_negative_sentiment(feedbacks)
        return summary


    # Function to process feedback for all schemes
    def process_feedbacks(data):
        # Group feedbacks by scheme
        grouped_feedbacks = data.groupby('scheme')

        # Initialize an empty dictionary to store summaries for each scheme
        scheme_summaries = {}

        # Iterate over each scheme and its corresponding feedbacks
        for scheme, feedbacks in grouped_feedbacks:
            # Process feedback for the current scheme
            summary = process_feedback_for_scheme(scheme, feedbacks.to_dict('records'))
            # Store the summary for the current scheme in the dictionary
            scheme_summaries[scheme] = summary

        return scheme_summaries


    # Function to read feedback data from a CSV file
    def read_feedback_data_from_csv(file_path):
        try:
            # Read feedback data from CSV
            feedback_data = pd.read_csv(file_path)
            return feedback_data
        except Exception as e:
            print("Error reading CSV file:", e)
            return None


    # Function to get CSV file path
    def get_csv_file_path():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "hackfest.feedback.csv")
        return file_path


    # Get CSV file path
    csv_file_path = get_csv_file_path()

    # Read feedback data from CSV
    feedback_data = read_feedback_data_from_csv(csv_file_path)

    # Process feedbacks if data is successfully read
    if feedback_data is not None:
        summarized_paragraph1 = process_feedbacks(feedback_data)

    print(summarized_paragraph1)

    # Initialize an empty list to store paragraphs with separators
    data_with_separators = []

    # Iterate over the items in summarized_paragraph1
    from reportlab.platypus import Paragraph, Spacer

    # Initialize an empty list to store paragraphs with separators
    data_with_separators = []

    for scheme, summary in summarized_paragraph1.items():
        # Create a Paragraph object for the scheme name
        scheme_name_paragraph = Paragraph("<b>Scheme:</b> {}".format(scheme))
        # Create a Paragraph object for the summary
        summary_paragraph = Paragraph(summary)
        # Add the scheme name paragraph to the list
        data_with_separators.append(scheme_name_paragraph)
        # Add the summary paragraph to the list
        data_with_separators.append(summary_paragraph)
        # Add a separator line
        data_with_separators.append(Spacer(1, 12))  # Add space between paragraphs

    # Remove the last spacer
    data_with_separators.pop()

    print(data_with_separators)


    # Create PDF report
    doc = SimpleDocTemplate("analysis_report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph("Transportation Analysis Report", styles['Title'])
    title2 = Paragraph("Nutrition Analysis Report", styles['Title'])
    title3 = Paragraph("Vaccination Analysis Report", styles['Title'])
    title4 = Paragraph("Summary of negatively polarized feedbacks", styles['Title'])

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
            Paragraph(
                "<b>Nutritional Consumption of Fruits by Income Group:</b><br/>"),
            Image('nutritional_consumption_fruits.png', width=300, height=200)
        ],
        [
            Paragraph(
                "<b>Nutritional Consumption of Vegetables by Income Group:</b><br/>"
            ),
            Image('nutritional_consumption_vegetables.png', width=300, height=200)
        ],
        [
            Paragraph("<b>Food Consumption:</b><br/>"
                    "{}".format(conclusion), styles['Normal']),
            Image('consumption_graph.png', width=300, height=200)
        ],
        [
            Paragraph(
                "<b>Clean Water Consumption:</b><br/>"
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

    # Create a list to hold the flowables
    data_summary = []

    # Add title for the summary
    data_summary.append(title4)

    # Iterate through data_with_separators and add paragraphs with separators
    for item in data_with_separators:
        # Check if the item is a string or a Paragraph object
        if isinstance(item, str):
            # If it's a string, add it as it is
            data_summary.append(Paragraph(item, styles['Normal']))
        elif isinstance(item, Paragraph):
            # If it's a Paragraph object, add it to the list
            data_summary.append(item)
        else:
            # If it's neither a string nor a Paragraph, skip it
            pass

        # Add a spacer for separation
        data_summary.append(Spacer(1, 12))

    # Remove the last spacer
    data_summary.pop()


    # Build the PDF report step by step

    # Create tables
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
    doc.build([title, table, PageBreak(), title2, table2, PageBreak(), title3, table3, PageBreak()] + data_summary)
    return 'analysis_report.pdf'

if __name__ == '__main__':
    app.run(debug=True)

