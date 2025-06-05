from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import torch
import rasterio
import json
import os

# Import Plotly with debug
try:
    import plotly
    import plotly.express as px
    import plotly.graph_objects as go
    print("Plotly imported successfully, version:", plotly.__version__)  # Fixed
except ImportError as e:
    print("Failed to import Plotly:", str(e))
    px = None
    go = None

from model import AODModel

app = Flask(__name__)

# Load data
try:
    df = pd.read_csv("data/aod1.csv")
    df.columns = df.columns.str.strip().str.title()
    df['Region'] = df['Region'].fillna('Unknown')
    df['Region'] = df['Region'].astype(str)
except FileNotFoundError:
    print("Error: data/aod1.csv not found")
    df = pd.DataFrame()

# Home
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/region_analysis', methods=['GET', 'POST'])
def region_analysis():
    if px is None:
        return render_template('region_analysis.html', error="Plotly not installed. Run 'pip install plotly==5.24.1'")

    try:
        regions = sorted(df['Region'].unique())
        years = sorted(df['Year'].unique())
        months = sorted(df['Month'].unique())
        df_valid = df[df['Aod'].notna()].copy()

        if request.method == 'POST':
            mode = request.form.get('mode')
            selected_region = request.form.get('region')

            if mode == 'region':
                yearly_avg = df_valid[df_valid['Region'] == selected_region].groupby('Year')['Aod'].mean().reset_index()
                monthly_avg = df_valid[df_valid['Region'] == selected_region].groupby('Month')['Aod'].mean().reset_index()

                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_avg['Month'] = pd.Categorical(monthly_avg['Month'], categories=month_order, ordered=True)
                monthly_avg = monthly_avg.sort_values('Month')

                yearly_fig = px.bar(yearly_avg, x='Year', y='Aod', title=f'AOD Trend by Year in {selected_region}', color='Aod', color_continuous_scale='Tealgrn')
                monthly_fig = px.bar(monthly_avg, x='Month', y='Aod', title=f'AOD Trend by Month in {selected_region}', color='Aod', color_continuous_scale='OrRd')

                max_month = monthly_avg.loc[monthly_avg['Aod'].idxmax()]
                min_month = monthly_avg.loc[monthly_avg['Aod'].idxmin()]
                high_month = f"Highest AOD: {max_month['Month']} ({max_month['Aod']:.4f})"
                low_month = f"Lowest AOD: {min_month['Month']} ({min_month['Aod']:.4f})"

                trend_text = "Insufficient data for trend analysis"
                if len(yearly_avg) > 1:
                    trend = np.polyfit(yearly_avg['Year'], yearly_avg['Aod'], 1)[0]
                    trend_text = f"AOD trend: {'Increasing' if trend > 0 else 'Decreasing' if trend < 0 else 'Stable'} year by year"

                monthly_highs = df_valid.groupby(['Month', 'Region'])['Aod'].mean().reset_index()
                monthly_highs = monthly_highs.loc[monthly_highs.groupby('Month')['Aod'].idxmax()]
                monthly_highs['Month'] = pd.Categorical(monthly_highs['Month'], categories=month_order, ordered=True)
                monthly_highs = monthly_highs.sort_values('Month').round(4)

                return render_template('region_analysis.html',
                                       regions=regions, years=years, months=months,
                                       selected_region=selected_region,
                                       yearly_json=json.dumps(yearly_fig, cls=plotly.utils.PlotlyJSONEncoder),
                                       monthly_json=json.dumps(monthly_fig, cls=plotly.utils.PlotlyJSONEncoder),
                                       high_month=high_month, low_month=low_month,
                                       trend_text=trend_text,
                                       monthly_highs=monthly_highs.to_dict('records'))

            elif mode == 'year':
                # When a specific year is selected
                year = int(request.form.get('year'))
                year_data = df_valid[(df_valid['Region'] == selected_region) & (df_valid['Year'] == year)]
                if year_data.empty:
                    return render_template('region_analysis.html', regions=regions, years=years, months=months,
                                           error=f"No data for {selected_region} in {year}")

                # Show AOD for all months in the selected year
                summary_df = year_data.groupby('Month')['Aod'].mean().reset_index().round(4)
                month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                summary_df['Month'] = pd.Categorical(summary_df['Month'], categories=month_order, ordered=True)
                summary_df = summary_df.sort_values('Month')

                summary_fig = px.bar(summary_df, x='Month', y='Aod',
                                     title=f'AOD in {year} for {selected_region}', color='Aod', color_continuous_scale='OrRd')

                top = summary_df.sort_values('Aod', ascending=False).iloc[0]
                summary_text = f"In {year}, the month with highest AOD was {top['Month']} ({top['Aod']:.4f})"

                return render_template('region_analysis.html',
                                       regions=regions, years=years, months=months,
                                       selected_region=selected_region,
                                       monthly_json=json.dumps(summary_fig, cls=plotly.utils.PlotlyJSONEncoder),
                                       trend_text=summary_text)

            elif mode == 'month':
                # When a specific month is selected
                month = request.form.get('month')
                month_data = df_valid[(df_valid['Region'] == selected_region) & (df_valid['Month'] == month)]
                if month_data.empty:
                    return render_template('region_analysis.html', regions=regions, years=years, months=months,
                                           error=f"No data for {selected_region} in {month}")

                # Show AOD for all years in the selected month
                summary_df = month_data.groupby('Year')['Aod'].mean().reset_index().round(4)
                summary_fig = px.bar(summary_df, x='Year', y='Aod',
                                     title=f'AOD in {month} for {selected_region}', color='Aod', color_continuous_scale='Bluered')

                top = summary_df.sort_values('Aod', ascending=False).iloc[0]
                summary_text = f"In {month}, the year with highest AOD was {top['Year']} ({top['Aod']:.4f})"

                return render_template('region_analysis.html',
                                       regions=regions, years=years, months=months,
                                       selected_region=selected_region,
                                       yearly_json=json.dumps(summary_fig, cls=plotly.utils.PlotlyJSONEncoder),
                                       trend_text=summary_text)

        return render_template('region_analysis.html', regions=regions, years=years, months=months)

    except Exception as e:
        return render_template('region_analysis.html', error=f"Error: {str(e)}")



# Predict
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.tif'):
            try:
                temp_path = 'temp.tif'
                file.save(temp_path)
                with rasterio.open(temp_path) as src:
                    image = src.read()
                if image.shape[0] < 13:
                    image = np.pad(image, ((0, 13 - image.shape[0]), (0, 0), (0, 0)), mode='constant')
                image_tensor = torch.tensor(image[:13], dtype=torch.float32).unsqueeze(0)
                model = AODModel()
                try:
                    state_dict = torch.load("models/aod_model.pth", map_location=torch.device("cpu"))
                    model.load_state_dict(state_dict)
                except RuntimeError as e:
                    os.remove(temp_path)
                    return render_template('predict.html', error=f"Model loading error: {str(e)}")
                model.eval()
                with torch.no_grad():
                    prediction = model(image_tensor).item()
                os.remove(temp_path)
                if prediction < 0.3:
                    quality = "Good"
                    alert = "success"
                elif prediction < 0.6:
                    quality = "Moderate"
                    alert = "warning"
                elif prediction < 0.9:
                    quality = "Unhealthy"
                    alert = "danger"
                else:
                    quality = "Hazardous"
                    alert = "danger"
                return render_template('predict.html', prediction=f"{prediction:.4f}", quality=quality, alert=alert)
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return render_template('predict.html', error=f"Prediction error: {str(e)}")
        return render_template('predict.html', error="Please upload a valid .tif file")
    return render_template('predict.html')

# Vizag hub
@app.route('/vizag')
def vizag():
    return render_template('vizag.html')



# Summary
@app.route('/summary', methods=['GET', 'POST'])
def summary():
    if px is None:
        return render_template('summary.html', error="Plotly not installed. Run 'pip install plotly==5.24.1'")
    
    try:
        regions = sorted(df['Region'].unique())
        print("Regions:", regions)  # Debug
        
        if request.method == 'POST':
            selected_region = request.form.get('region')
            print("Selected region:", selected_region)  # Debug
            if not selected_region:
                return render_template('summary.html', regions=regions, error="Please select a Region")
            
            df_valid = df[df['Aod'].notna()].copy()
            print("Valid rows:", len(df_valid))  # Debug
            
            # Yearly average
            print("Computing yearly avg...")  # Debug
            yearly_avg = df_valid[df_valid['Region'] == selected_region].groupby('Year')['Aod'].mean().reset_index()
            print("Yearly avg:", yearly_avg.to_dict())  # Debug
            yearly_fig = px.bar(
                yearly_avg,
                x='Year',
                y='Aod',
                title=f'Average AOD per Year in {selected_region}',
                color='Aod',
                color_continuous_scale='Tealgrn'
            )
            yearly_json = json.dumps(yearly_fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # Monthly average
            print("Computing monthly avg...")  # Debug
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_avg = df_valid[df_valid['Region'] == selected_region].groupby('Month')['Aod'].mean().reset_index()
            monthly_avg['Month'] = pd.Categorical(monthly_avg['Month'], categories=month_order, ordered=True)
            monthly_avg = monthly_avg.sort_values('Month')
            print("Monthly avg:", monthly_avg.to_dict())  # Debug
            monthly_fig = px.bar(
                monthly_avg,
                x='Month',
                y='Aod',
                title=f'Average AOD per Month in {selected_region}',
                color='Aod',
                color_continuous_scale='OrRd'
            )
            monthly_json = json.dumps(monthly_fig, cls=plotly.utils.PlotlyJSONEncoder)
            
            # High/low months
            high_month = low_month = "No data available"
            if not monthly_avg.empty:
                max_month = monthly_avg.loc[monthly_avg['Aod'].idxmax()]
                min_month = monthly_avg.loc[monthly_avg['Aod'].idxmin()]
                high_month = f"Highest AOD: {max_month['Month']} ({max_month['Aod']:.4f})"
                low_month = f"Lowest AOD: {min_month['Month']} ({min_month['Aod']:.4f})"
            
            # Trend
            trend_text = "Insufficient data for trend analysis"
            if len(yearly_avg) > 1:
                trend = np.polyfit(yearly_avg['Year'], yearly_avg['Aod'], 1)[0]
                trend_text = "Increasing" if trend > 0 else "Decreasing" if trend < 0 else "Stable"
                trend_text = f"AOD trend: {trend_text} year by year"
            
            # Monthly highs
            print("Computing monthly highs...")  # Debug
            monthly_highs = df_valid.groupby(['Month', 'Region'])['Aod'].mean().reset_index()
            monthly_highs = monthly_highs.loc[monthly_highs.groupby('Month')['Aod'].idxmax()]
            monthly_highs['Month'] = pd.Categorical(monthly_highs['Month'], categories=month_order, ordered=True)
            monthly_highs = monthly_highs.sort_values('Month').round(4)
            print("Monthly highs:", monthly_highs.to_dict())  # Debug
            
            return render_template('summary.html',
                                 regions=regions,
                                 selected_region=selected_region,
                                 yearly_json=yearly_json,
                                 monthly_json=monthly_json,
                                 high_month=high_month,
                                 low_month=low_month,
                                 trend_text=trend_text,
                                 monthly_highs=monthly_highs.to_dict('records'))
        
        return render_template('summary.html', regions=regions)
    
    except Exception as e:
        print("Summary error:", str(e))  # Debug
        return render_template('summary.html', regions=regions, error=f"Error: {str(e)}")

# Bar chart
@app.route('/bar_chart', methods=['GET', 'POST'])
def bar_chart():
    if px is None:
        return render_template('bar_chart.html', error="Plotly not installed. Run 'pip install plotly==5.24.1'")
    
    try:
        years = sorted(df['Year'].unique())
        months = sorted(df['Month'].unique())
        
        if request.method == 'POST':
            selected_years = request.form.getlist('years')
            selected_months = request.form.getlist('months')
            if not selected_years:
                selected_years = years
            if not selected_months:
                selected_months = months
            selected_years = [int(y) for y in selected_years]
        else:
            selected_years = years
            selected_months = months
        
        df_valid = df[df['Aod'].notna() & df['Year'].isin(selected_years) & df['Month'].isin(selected_months)]
        if df_valid.empty:
            return render_template('bar_chart.html', years=years, months=months, error="No data for selected filters")
        
        bar_df = df_valid.groupby('Region')['Aod'].mean().reset_index().round(4)
        fig = px.bar(
            bar_df,
            x='Region',
            y='Aod',
            color='Aod',
            color_continuous_scale='Tealgrn',
            title='Average AOD per Region'
        )
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('bar_chart.html',
                             years=years,
                             months=months,
                             selected_years=selected_years,
                             selected_months=selected_months,
                             graph_json=graph_json)
    except Exception as e:
        return render_template('bar_chart.html', years=years, months=months, error=f"Error: {str(e)}")

# Map view
@app.route('/map_view', methods=['GET', 'POST'])
def map_view():
    if px is None:
        return render_template('map_view.html', error="Plotly not installed. Run 'pip install plotly==5.24.1'")
    
    try:
        years = sorted(df['Year'].unique())
        months = sorted(df['Month'].unique())
        
        if request.method == 'POST':
            selected_years = request.form.getlist('years')
            selected_months = request.form.getlist('months')
            if not selected_years:
                selected_years = years
            if not selected_months:
                selected_months = months
            selected_years = [int(y) for y in selected_years]
        else:
            selected_years = years
            selected_months = months
        
        df_valid = df[df['Aod'].notna() & df['Year'].isin(selected_years) & df['Month'].isin(selected_months)]
        if df_valid.empty:
            return render_template('map_view.html', years=years, months=months, error="No data for selected filters")
        
        fig = px.scatter_mapbox(
            df_valid,
            lat='Latitude',
            lon='Longitude',
            color='Aod',
            size='Aod',
            hover_name='Region',
            hover_data={'Aod': ':.4f'},
            color_continuous_scale='OrRd',
            zoom=11,
            height=600,
            title='Geographic Distribution of AOD'
        )
        fig.update_layout(mapbox_style='carto-positron', margin={'r':0, 't':50, 'l':0, 'b':0})
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('map_view.html',
                             years=years,
                             months=months,
                             selected_years=selected_years,
                             selected_months=selected_months,
                             graph_json=graph_json)
    except Exception as e:
        return render_template('map_view.html', years=years, months=months, error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)