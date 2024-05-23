from flask import Flask, redirect, render_template, request, url_for
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['hackfest']
collection3 = db['vaccination']
collection4=db['transportation']
collection5= db['nutritionForm']
collection6= db['Pwomen']
collection7 = db['Gscheme']

@app.route('/')
def index():
    # Get today's date
    today_date = datetime.now().strftime('%Y-%m-%d')
    # Retrieve today's data if exists
    survey_data = collection4.find_one({'date': today_date})
    return render_template('ALL.html', today_date=today_date, survey_data=survey_data)


@app.route('/submit_vaccination', methods=['POST'])
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
    collection3.insert_one(data)

    return redirect(url_for('index'))


@app.route('/submit_transportation', methods=['POST'])
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
        collection4.insert_one(survey_data)

        return redirect(url_for('index'))
    

@app.route('/submit_nutrition', methods=['POST'])
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
        collection5.insert_one(nutrition_data)

        return redirect(url_for('index'))
    
@app.route('/submit_pregnant', methods=['POST'])
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
        collection6.insert_one(pregnant_data)

        return redirect(url_for('index'))
    
@app.route('/submit_benefits', methods=['POST'])
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

        return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True)
