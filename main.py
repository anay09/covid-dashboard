from flask import Flask, render_template, request, session
import os
import traceback
from matplotlib import pyplot as plt
import io
import base64
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import iplot
from scipy.stats import chi2_contingency
from scipy.stats import chi2
import seaborn as sn


app = Flask(__name__)
app.secret_key = "password"


def multi_plot_line(df, filename, graph_title, x_title, addAll = True):
	fig = go.Figure()

	for column in df.columns.to_list():
		fig.add_trace(
			go.Scatter(
				x = df.index,
				y = df[column],
				name = column
			)
		)

	button_all = dict(label = 'All',
					  method = 'update',
					  args = [{'visible': df.columns.isin(df.columns),
							   'title': 'All',
							   'showlegend':True}])

	def create_layout_button(column):
		return dict(label = column,
					method = 'update',
					args = [{'visible': df.columns.isin([column]),
							 'title': column,
							 'showlegend': True}])

	fig.update_layout(
		updatemenus=[go.layout.Updatemenu(
			active = 0,
			buttons = ([button_all] * addAll) + list(df.columns.map(lambda column: create_layout_button(column)))
			)
		],
		paper_bgcolor='rgba(20,20,20,1)',
		title = graph_title,
		xaxis= dict(title= x_title,ticklen= 5,zeroline= False),
		font=dict(
			family="Arial",
			size=15,
			color="rgb(150, 150, 150)"
			)
	)

	new_filename = "templates/" + filename
	fig.write_html(new_filename)


def multi_stacked_plot_line(dfs, subjects, filenames, graph_title, x_title):
	fig = [ None for i in range(len(dfs[0].columns.to_list()))]

	for ind, column in enumerate(dfs[0].columns.to_list()):
		fig[ind] = go.Figure()

		for ind_sub, sub in enumerate(subjects):
			fig[ind].add_trace(
				go.Scatter(
					x = dfs[ind_sub].index,
					y = dfs[ind_sub][column],
					mode='lines',
					name = sub
				)
			)

		fig[ind].update_layout(
			paper_bgcolor='rgba(20,20,20,1)',
			title = graph_title + filenames[ind].split(".")[0].upper(),
			xaxis=dict(title=x_title,ticklen= 5,zeroline= False),
			font=dict(
				family="Arial",
				size=15,
				color="rgb(180, 180, 180)"
			)
		)

		fig[ind].write_html("templates/" + filenames[ind])


def multi_plot_bar(df, filename, graph_title, x_title, addAll = True):
	fig = go.Figure()

	for column in df.columns.to_list():
		fig.add_trace(
			go.Bar(
				x = df.index,
				y = df[column],
				name = column
			)
		)

	button_all = dict(label = 'All',
					  method = 'update',
					  args = [{'visible': df.columns.isin(df.columns),
							   'title': 'All',
							   'showlegend':True}])

	def create_layout_button(column):
		return dict(label = column,
					method = 'update',
					args = [{'visible': df.columns.isin([column]),
							 'title': column,
							 'showlegend': True}])

	fig.update_layout(
		updatemenus=[go.layout.Updatemenu(
			active = 0,
			buttons = ([button_all] * addAll) + list(df.columns.map(lambda column: create_layout_button(column)))
			)
		],
		paper_bgcolor='rgba(20,20,20,1)',
		title = graph_title,
		xaxis= dict(title= x_title,ticklen= 5,zeroline= False),
		font=dict(
			family="Arial",
			size=15,
			color="rgb(150, 150, 150)"
			)
	)

	new_filename = "templates/" + filename
	fig.write_html(new_filename)



def multi_plot_scatter(df, filename, graph_title, x_title, addAll = True):
	fig = go.Figure()

	colors = ['rgba(192, 0, 0, 1)', 'rgba(50, 52, 170, 1)', 'rgba(155, 102, 193, .7)']

	for ind, column in enumerate(df.columns.to_list()):
		fig.add_trace(
			go.Scatter(
				x = df.index,
				y = df[column],
				name = column,
				mode='markers',
				marker_color=colors[ind]
			)
		)

	fig.update_traces(mode='markers', marker_line_width=0, marker_size=15, marker_opacity=0.8)
	fig.update_layout(
		title=graph_title, 
		yaxis_zeroline=False, 
		xaxis_zeroline=False,
		paper_bgcolor='rgba(20, 20, 20, 1)',
		xaxis=dict(title= x_title, ticklen= 5, zeroline= False),
		font=dict(
			family="Arial",
			size=15,
			color="rgb(200, 200, 200)"
		)
	)

	# fig.write_image("demographic_by_gender.svg")

	new_filename = "templates/" + filename
	fig.write_html(new_filename)



@app.route("/", methods = ["GET"])
def index_page():
	dates = ['01/22/20', '01/23/20', '01/24/20', '01/25/20', '01/26/20', '01/27/20', '01/28/20', '01/29/20', '01/30/20', '01/31/20', '02/01/20', '02/02/20', '02/03/20', '02/04/20', '02/05/20', '02/06/20', '02/07/20', '02/08/20', '02/09/20', '02/10/20', '02/11/20', '02/12/20', '02/13/20', '02/14/20', '02/15/20', '02/16/20', '02/17/20', '02/18/20', '02/19/20', '02/20/20', '02/21/20', '02/22/20', '02/23/20', '02/24/20', '02/25/20', '02/26/20', '02/27/20', '02/28/20', '02/29/20', '03/01/20', '03/02/20', '03/03/20', '03/04/20', '03/05/20', '03/06/20', '03/07/20', '03/08/20', '03/09/20', '03/10/20', '03/11/20', '03/12/20', '03/13/20', '03/14/20', '03/15/20', '03/16/20', '03/17/20', '03/18/20', '03/19/20', '03/20/20', '03/21/20', '03/22/20', '03/23/20', '03/24/20', '03/25/20', '03/26/20', '03/27/20', '03/28/20', '03/29/20', '03/30/20', '03/31/20', '04/01/20', '04/02/20', '04/03/20', '04/04/20', '04/05/20', '04/06/20', '04/07/20', '04/08/20', '04/09/20', '04/10/20', '04/11/20', '04/12/20', '04/13/20', '04/14/20', '04/15/20', '04/16/20', '04/17/20', '04/18/20', '04/19/20', '04/20/20', '04/21/20', '04/22/20', '04/23/20', '04/24/20', '04/25/20', '04/26/20', '04/27/20', '04/28/20', '04/29/20', '04/30/20', '05/01/20', '05/02/20', '05/03/20', '05/04/20', '05/05/20', '05/06/20', '05/07/20', '05/08/20', '05/09/20', '05/10/20', '05/11/20', '05/12/20', '05/13/20', '05/14/20', '05/15/20', '05/16/20', '05/17/20', '05/18/20', '05/19/20', '05/20/20', '05/21/20', '05/22/20', '05/23/20', '05/24/20', '05/25/20', '05/26/20', '05/27/20', '05/28/20', '05/29/20', '05/30/20', '05/31/20', '06/01/20']
	countries = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burma', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo (Brazzaville)', 'Congo (Kinshasa)', 'Costa Rica',  'Croatia', 'Cuba', 'Cyprus', 'Czechia', 'Denmark', 'Diamond Princess', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Holy See', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Korea, South', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'MS Zaandam', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Somalia', 'South Africa', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syria', 'Taiwan*', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'US', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'Uruguay', 'Uzbekistan', 'Venezuela', 'Vietnam', 'West Bank and Gaza', 'Western Sahara', 'Yemen', 'Zambia', 'Zimbabwe']
	
	df = pd.read_csv('Datasets/countries-aggregated.csv')
	new_df = df.groupby(['Country']).max()
	new_df = new_df.sort_values(by=['Confirmed'], ascending = False).drop(['Date'],axis=1)
	# final_df = new_df.head(20)
	final_df = new_df
	c_list = final_df.index

	return render_template('index.html', column_names = final_df.columns.values, row_data = list(final_df.values.tolist()), country = c_list, zip=zip)


@app.route("/time_series", methods = ["GET", "POST"])
def time_series():
	dates = ['01/22/20', '01/23/20', '01/24/20', '01/25/20', '01/26/20', '01/27/20', '01/28/20', '01/29/20', '01/30/20', '01/31/20', '02/01/20', '02/02/20', '02/03/20', '02/04/20', '02/05/20', '02/06/20', '02/07/20', '02/08/20', '02/09/20', '02/10/20', '02/11/20', '02/12/20', '02/13/20', '02/14/20', '02/15/20', '02/16/20', '02/17/20', '02/18/20', '02/19/20', '02/20/20', '02/21/20', '02/22/20', '02/23/20', '02/24/20', '02/25/20', '02/26/20', '02/27/20', '02/28/20', '02/29/20', '03/01/20', '03/02/20', '03/03/20', '03/04/20', '03/05/20', '03/06/20', '03/07/20', '03/08/20', '03/09/20', '03/10/20', '03/11/20', '03/12/20', '03/13/20', '03/14/20', '03/15/20', '03/16/20', '03/17/20', '03/18/20', '03/19/20', '03/20/20', '03/21/20', '03/22/20', '03/23/20', '03/24/20', '03/25/20', '03/26/20', '03/27/20', '03/28/20', '03/29/20', '03/30/20', '03/31/20', '04/01/20', '04/02/20', '04/03/20', '04/04/20', '04/05/20', '04/06/20', '04/07/20', '04/08/20', '04/09/20', '04/10/20', '04/11/20', '04/12/20', '04/13/20', '04/14/20', '04/15/20', '04/16/20', '04/17/20', '04/18/20', '04/19/20', '04/20/20', '04/21/20', '04/22/20', '04/23/20', '04/24/20', '04/25/20', '04/26/20', '04/27/20', '04/28/20', '04/29/20', '04/30/20', '05/01/20', '05/02/20', '05/03/20', '05/04/20', '05/05/20', '05/06/20', '05/07/20', '05/08/20', '05/09/20', '05/10/20', '05/11/20', '05/12/20', '05/13/20', '05/14/20', '05/15/20', '05/16/20', '05/17/20', '05/18/20', '05/19/20', '05/20/20', '05/21/20', '05/22/20', '05/23/20', '05/24/20', '05/25/20', '05/26/20', '05/27/20', '05/28/20', '05/29/20', '05/30/20', '05/31/20', '06/01/20']
	countries = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Burma', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo (Brazzaville)', 'Congo (Kinshasa)', 'Costa Rica',  'Croatia', 'Cuba', 'Cyprus', 'Czechia', 'Denmark', 'Diamond Princess', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Holy See', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Korea, South', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'MS Zaandam', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Somalia', 'South Africa', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Syria', 'Taiwan*', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'US', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'Uruguay', 'Uzbekistan', 'Venezuela', 'Vietnam', 'West Bank and Gaza', 'Western Sahara', 'Yemen', 'Zambia', 'Zimbabwe']
	return render_template('time_series.html', dates=dates, countries=countries)



@app.route("/prediction", methods = ["GET", "POST"])
def prediction():
	countries = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Argentina', 'Armenia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Benin', 'Bolivia', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Chad', 'Chile', 'Colombia', 'Congo (Brazzaville)', 'Congo (Kinshasa)', 'Costa Rica',  'Cuba', 'Cyprus', 'Czechia', 'Djibouti',  'Ecuador', 'Egypt', 'El Salvador', 'Estonia', 'Ethiopia', 'Finland', 'Gabon', 'Georgia', 'Germany', 'Greece','Guatemala', 'Guinea',  'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kuwait', 'Kyrgyzstan', 'Latvia', 'Lebanon', 'Liberia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malaysia','Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco',  'Nepal', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Norway', 'Oman', 'Pakistan', 'Panama', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'San Marino', 'Saudi Arabia', 'Senegal', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'Somalia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Sweden', 'Switzerland', 'Taiwan', 'Tanzania', 'Thailand', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'United States of America', 'Uganda', 'Ukraine', 'United Arab Emirates' ]
	return render_template('prediction.html',countries = countries)


@app.route("/prediction_country", methods = ["GET", "POST"])
def prediction_country():
	if request.method == 'POST':
		result = request.form
	country = result['country']
	df = pd.read_csv('Datasets/prediction_data.csv')
	df = df.set_index('Dates')
	new_df = df[country]
	data = go.Scatter(x = new_df.index, y=new_df.values, mode="lines")
	layout = dict(title = country.upper() + ' : Prediction of cases for next 10 days', xaxis= dict(title= 'Date',ticklen= 5,zeroline= False), yaxis = dict(title= 'No of cases'),paper_bgcolor='rgba(20,20,20,1)', font=dict(family="Arial",size=15,color="rgb(200, 200, 200)"))
	fig = go.Figure(data = data, layout = layout)
	new_filename = "templates/" + "country_pred.html"
	fig.write_html(new_filename)
	return render_template('prediction_op.html',filename="country_pred.html")




@app.route("/correlation", methods = ["GET","POST"])
def correlation_page():
	attributes = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths',
	   'total_cases_per_million', 'new_cases_per_million',
	   'total_deaths_per_million', 'new_deaths_per_million',
	   'stringency_index', 'population', 'population_density', 'median_age',
	   'aged_65_older', 'aged_70_older', 'gdp_per_capita', 'cvd_death_rate',
	   'diabetes_prevalence', 'female_smokers', 'male_smokers',
	   'handwashing_facilities', 'hospital_beds_per_100k']
	return render_template('correlation.html', correlation="true", attributes=attributes)


@app.route("/correlation_heatmap", methods = ["GET","POST"])
def heatmap():
	feature_df = pd.read_csv('Datasets/owid-covid-data.csv')
	feature_df = feature_df.drop(['iso_code','date','total_tests','new_tests','total_tests_per_thousand','new_tests_per_thousand','new_tests_smoothed','new_tests_smoothed_per_thousand','tests_units','extreme_poverty'],axis=1)
	feature_df = feature_df.replace(np.nan,0)
	feature_df = feature_df.groupby(['location']).max()
	colorscale = [[0, '#edf8fb'], [.3, '#b3cde3'],  [.6, '#8856a7'],  [1, '#810f7c']]

	heatmap = go.Heatmap(z=feature_df.corr(), x=feature_df.columns, y=feature_df.columns, colorscale=colorscale)
	data = [heatmap]
	fig = go.Figure(data = data)

	fig.update_layout(
		title="Heatmap (Correlation b/w all possible pairs of attributes) : ", 
		yaxis_zeroline=False, 
		xaxis_zeroline=False,
		paper_bgcolor='rgba(30, 30, 30, 1)',
		# xaxis=dict(title= x_title, ticklen= 5, zeroline= False),
		font=dict(
			family="Arial",
			size=13,
			color="wheat"
		)
	)

	new_filename = "templates/" + "heatmap.html"
	fig.write_html(new_filename)

	return render_template('correlation.html', heatmap="true", filename="heatmap.html")


@app.route("/corr_coef", methods = ["GET","POST"])
def corr_coefficient():
	feature_df = pd.read_csv('Datasets/owid-covid-data.csv')
	feature_df.columns
	feature_df = feature_df.drop(['iso_code','date','total_tests','new_tests','total_tests_per_thousand','new_tests_per_thousand','new_tests_smoothed','new_tests_smoothed_per_thousand','tests_units','extreme_poverty'],axis=1)
	feature_df = feature_df.replace(np.nan,0)
	feature_df = feature_df.groupby(['location']).max()
	attributes = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths',
	   'total_cases_per_million', 'new_cases_per_million',
	   'total_deaths_per_million', 'new_deaths_per_million',
	   'stringency_index', 'population', 'population_density', 'median_age',
	   'aged_65_older', 'aged_70_older', 'gdp_per_capita', 'cvd_death_rate',
	   'diabetes_prevalence', 'female_smokers', 'male_smokers',
	   'handwashing_facilities', 'hospital_beds_per_100k']
	if request.method == 'POST':
		result = request.form 
	column1 = feature_df[result['attribute1']]
	column2 = feature_df[result['attribute2']]
	correlation = column1.corr(column2)
	return render_template('correlation.html', attributes=attributes, coefficient=correlation, correlation="true")


@app.route("/timeseries", methods = ['POST','GET'])
def timeseries_world():
	df = pd.read_csv('Datasets/worldwide-aggregated.csv')
	world_df = df.groupby(['Date']).sum()
	multi_plot_line(world_df, "world_plot.html", "Worldwide Covid Cases", "Date")
	return render_template('time_series_op.html',filename='world_plot.html')


@app.route("/timeseries_country", methods = ['POST','GET'])
def timeseries_country():
	if request.method == 'POST':
		result = request.form
	country = result['country']
	df = pd.read_csv('Datasets/countries-aggregated.csv')
	new_df = df.groupby(['Country','Date']).max()
	new_df = new_df.T
	temp = new_df[country].T
	multi_plot_line(temp, "country_plot.html", "Covid Cases in " + country, "Date")
	return render_template('time_series_op.html',filename='country_plot.html')


@app.route("/compare", methods = ['POST','GET'])
def timeseries_compare():
	if request.method == 'POST':
		result = request.form
	country1 = result['country1']
	country2 = result['country2']
	df = pd.read_csv('Datasets/countries-aggregated.csv')
	new_df = df.groupby(['Country','Date']).max()
	new_df = new_df.T
	temp1 = new_df[country1].T
	# multi_plot_line(temp1, "country_plot1.html", "Covid Cases in " + country1, "Date")
	temp2 = new_df[country2].T
	# multi_plot_line(temp2, "country_plot2.html", "Covid Cases in " + country2, "Date")
	multi_stacked_plot_line([temp1, temp2], [country1, country2], ["confirmed.html", "recovered.html", "death.html", "active.html"], "COVID Cases : ", "Date")
	return render_template('compare_op.html', filename1="confirmed.html", country1=country1, country2=country2)


@app.route("/compare_recovered", methods = ['POST','GET'])
def compare_recovered():
	return render_template('recovered.html')

@app.route("/compare_death", methods = ['POST','GET'])
def compare_death():
	return render_template('death.html')

@app.route("/compare_active", methods = ['POST','GET'])
def compare_active():
	return render_template('active.html')

@app.route("/compare_confirmed", methods = ['POST','GET'])
def compare_confirmed():
	return render_template('confirmed.html')


@app.route("/timeseries_date", methods = ['POST','GET'])
def timeseries_date():
	if request.method == 'POST':
		result = request.form
	date = result['date']
	df = pd.read_csv('Datasets/countries-aggregated.csv')
	new_df = df.groupby(['Date','Country']).max()
	new_df = new_df.T
	temp = new_df[date].T
	multi_plot_line(temp, "date_plot.html", "Covid Cases upto date " + date, "Country")
	return render_template('time_series_op.html',filename='date_plot.html')



@app.route("/geographic", methods = ['POST','GET'])
def geographic_analysis():
	return render_template('geographic_op.html',filename='map_dark.html')


@app.route("/demographic", methods = ['POST','GET'])
def demographic_analysis():
	return render_template('demographic.html')

'''
@app.route("/age", methods = ['POST'])
def age_analysis():
	img_cnf = io.BytesIO()
	img_dec = io.BytesIO()
	

	if request.method == 'POST':
		result = request.form
	country = result['country']
	plot = result['plot']
	time_age = pd.read_csv('Datasets/coronavirusdataset/TimeAge.csv')
	time_age = time_age.drop(['time'],axis=1)
	cnf = time_age.drop(['deceased'],axis=1)
	dec = time_age.drop(['confirmed'],axis=1)
	#age = time_age.groupby(['age']).sum()
	#ge.plot(kind=plot)

	cnf.groupby('age').sum().plot(kind=plot)
	
	plt.savefig(img_cnf, format='png', bbox_inches = "tight")
	img_cnf.seek(0)
	plot_url_cnf = base64.b64encode(img_cnf.getvalue()).decode()
	
	dec.groupby('age').sum().plot(kind=plot)

	plt.savefig(img_dec, format='png', bbox_inches = "tight")
	img_dec.seek(0)
	plot_url_dec = base64.b64encode(img_dec.getvalue()).decode()
	

	plot_url = []
	plot_url.append(plot_url_cnf)
	plot_url.append(plot_url_dec)

	return render_template('age_analysis.html', plot_url = plot_url, country=country)
	#return '<img src="data:image/png;base64,{}">'.format(plot_url)
'''

@app.route("/age", methods = ['POST'])
def age_analysis():
	time_age = pd.read_csv('Datasets/TimeAge.csv')
	time_age = time_age.drop(['time'],axis=1)
	age = time_age.groupby(['age']).sum()
	multi_plot_bar(age,"age_plotly","Agewise analysis in South Korea", "Age groups")
	return render_template("age_plotly")
	



@app.route("/gender", methods = ['POST','GET'])
def gender_analysis():
	time_gender = pd.read_csv('Datasets/TimeGender.csv')
	time_gender = time_gender.drop(['time'],axis=1)
	gender = time_gender.groupby(['sex']).sum()
	multi_plot_bar(gender,"gender_plotly","Genderwise Analysis in South Korea","Gender")
	return render_template("gender_plotly")



@app.route("/hyptesting", methods = ['POST', 'GET'])
def hyptesting():
	return render_template('hypothesis.html')




@app.route("/hyptesting_age", methods = ['POST', 'GET'])
def hyptesting_age():
	read1 = pd.DataFrame(
	[
		[6404,2],
		[29183,5],
		[13249,7],
		[8621,72],
		[75018,112],
		[103480,622],
		[69622,1381],
		[35957,2764],
		[22981,4146]
	],
	index=["0s","10s","20s","30s","40s","50s","60s","70s","80s"],
	columns=["CONFIRMED","DEATHS"])
	time_age = pd.read_csv('Datasets/TimeAge.csv')
	time_age = time_age.drop(['time'],axis=1)
	age = time_age.groupby(['age']).sum()
	multi_plot_scatter(age,"age_plotly.html","Agewise analysis", "Age groups")
	filename = "age_plotly.html"
	chi, pval, dof, exp = chi2_contingency(read1)
	significance = 0.05
	p = 1 - significance
	critical_value = chi2.ppf(p, dof)
	#print('chi=%.6f, critical value=%.6f\n' % (chi, critical_value))
	if chi > critical_value:
		retval = """At %.2f level of significance, we reject the null hypotheses and accept Alternate Hypothesis. They are dependent.""" % (significance)
	else:
		retval = """At %.2f level of significance, we accept the null hypotheses. 
	They are independent.""" % (significance)

	return render_template('hypothesis.html', retval = retval, filename = filename)



if __name__ == "__main__":
	app.run(debug=True)
