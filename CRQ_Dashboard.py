import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image  # Python Imaging Library

# Load your image
image = Image.open("cedaf_ai2.png")  # or .png, .gif, etc.



st.set_page_config(layout="wide")


@st.cache_data  # Cache the data loading and calculations
def load_and_process_data():
    # Load your dataset (update path as needed)
    df = pd.read_excel("Synthetic_Cyber_Risk_Data_with_Exposure_Score.xlsx")
    
    # Compute expected annual and residual loss
    df['Expected Annual Loss ($)'] = (
        df['Frequency'] *
        df['Vulnerability Score'] *
        (df['Primary Loss ($)'] + df['Secondary Loss ($)'])
    )
    
    df['Residual Loss ($)'] = (
        df['Expected Annual Loss ($)'] *
        (1 - df['Control Maturity Score'])
    )
    
    # Monte Carlo CVaR Simulation
    n_simulations = 10000
    simulated_means = []
    simulated_cvars = []

    for _, row in df.iterrows():
        lambda_freq = row['Frequency']
        residual_loss = row['Residual Loss ($)']
        
        if lambda_freq == 0 or residual_loss == 0:
            simulated_means.append(0.0)
            simulated_cvars.append(0.0)
        else:
            avg_loss = residual_loss / lambda_freq
            mu = np.log(avg_loss + 1e-6)
            sigma = 0.8
            
            freq_sim = np.random.poisson(lam=lambda_freq, size=n_simulations)
            loss_per_event = np.random.lognormal(mean=mu, sigma=sigma, size=n_simulations)
            total_loss = freq_sim * loss_per_event
            
            simulated_means.append(np.mean(total_loss))
            tail = total_loss[total_loss >= np.percentile(total_loss, 95)]
            simulated_cvars.append(np.mean(tail))

    df['Simulated Mean Loss ($)'] = simulated_means
    df['Simulated CVaR 95% ($)'] = simulated_cvars
    
    return df

df = load_and_process_data()


def main():
    # st.title("Cyber Risk Insurance Executive Dashboard")
    #st.write("Cyber Risk Quantification Model")

    st.markdown("<h1 style='text-align: center; font-size: 60px;color: #002a6f ;'> Executive Dashboard</h1>", 
    unsafe_allow_html=True )
    st.markdown("<p style='text-align: center; font-size: 30px;'>Cyber Risk Insurance Quantification Model </p>", 
    unsafe_allow_html=True )

    # Display in sidebar
    st.sidebar.image(image)
    st.sidebar.markdown("<hr style='border:1px solid #002a6f '>", unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.header("Filter Data")
    
    # Get unique values for filters
    all_industries = ["All"] + sorted(df["Industry"].unique().tolist())
    all_vectors = ["All"] + sorted(df["Attack Vector"].unique().tolist())
    
    # Create dropdown filters
    selected_industry = st.sidebar.selectbox("Select Industry", all_industries)
    selected_attack = st.sidebar.selectbox("Select Attack Vector", all_vectors)
    
    # Create frequency slider
    min_freq, max_freq = int(df['Frequency'].min()), int(df['Frequency'].max())
    freq_range = st.sidebar.slider(
        "Select Frequency Range",
        min_value=min_freq,
        max_value=max_freq,
        value=(min_freq, max_freq)
    )
    



    # Filter data based on selections
    filtered_df = df.copy()
    if selected_industry != "All":
        filtered_df = filtered_df[filtered_df["Industry"] == selected_industry]
    if selected_attack != "All":
        filtered_df = filtered_df[filtered_df["Attack Vector"] == selected_attack]
    
    # Apply frequency filter
    filtered_df = filtered_df[
        (filtered_df['Frequency'] >= freq_range[0]) & 
        (filtered_df['Frequency'] <= freq_range[1])
    ]
    
    # Display filtered data stats
    st.sidebar.write(f" {len(filtered_df)} records matching your filters")
    
    # Create dashboard visualizations
    create_dashboard_visualizations(filtered_df)


def create_dashboard_visualizations(df):
    # Set style
    sns.set(style="whitegrid")
    
    # Determine top industries and vectors based on filtered data
    top_industries = df.groupby('Industry')['Simulated CVaR 95% ($)'].mean().nlargest(7).index
    df_top_industry = df[df['Industry'].isin(top_industries)]
    
    top_vectors = df.groupby('Attack Vector')['Simulated CVaR 95% ($)'].mean().nlargest(7).index
    df_top_vector = df[df['Attack Vector'].isin(top_vectors)]
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Risk Overview", "Exposure Analysis", "CVaR Heatmap", "Residual Risk and Portfolio Simulation"])
    
    with tab1:
        # First row of charts
        col1, col2 = st.columns(2)
        
        with col1:
            #st.subheader("Top Industries by Average CVaR")
            st.markdown( "<h1 style='font-size: 16px; text-align: center;'>Top Industries by Average CVaR</h1>", 
            unsafe_allow_html=True )
            industry_avg_cvar = df_top_industry.groupby('Industry')['Simulated CVaR 95% ($)'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=industry_avg_cvar.values, y=industry_avg_cvar.index, palette='Reds_r', ax=ax)
            ax.set_xlabel("Average CVaR 95% ($)")
            st.pyplot(fig)
        
        with col2:
            #st.subheader("Top Attack Vectors by Average CVaR")
            st.markdown( "<h1 style='font-size: 16px; text-align: center;'>Top Attack Vectors by Average CVaR</h1>", 
            unsafe_allow_html=True )
            vector_avg_cvar = df_top_vector.groupby('Attack Vector')['Simulated CVaR 95% ($)'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=vector_avg_cvar.values, y=vector_avg_cvar.index, palette='Oranges_r', ax=ax)
            ax.set_xlabel("Average CVaR 95% ($)")
            st.pyplot(fig)
        
        
    
    with tab2:
        # Exposure analysis charts
        col1, col2 = st.columns(2)
        
        with col1:
            #st.subheader("CVaR vs Risk Exposure by Industry")
            st.markdown( "<h1 style='font-size: 16px; text-align: center;'>CVaR vs Risk Exposure by Industry</h1>", 
            unsafe_allow_html=True )
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df_top_industry, x='Risk Exposure Score', y='Simulated CVaR 95% ($)', 
                           hue='Industry', alpha=0.8, ax=ax)
            ax.set_xlabel("Risk Exposure Score")
            ax.set_ylabel("Simulated CVaR 95% ($)")
            st.pyplot(fig)
        
        with col2:
            #st.subheader("CVaR vs Risk Exposure by Attack Vector")
            st.markdown( "<h1 style='font-size: 16px; text-align: center;'>CVaR vs Risk Exposure by Attack Vector</h1>", 
            unsafe_allow_html=True )
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df_top_vector, x='Risk Exposure Score', y='Simulated CVaR 95% ($)', 
                           hue='Attack Vector', alpha=0.8, ax=ax)
            ax.set_xlabel("Risk Exposure Score")
            ax.set_ylabel("Simulated CVaR 95% ($)")
            st.pyplot(fig)
    
    with tab3:
        # Heatmap
        #st.subheader("CVaR Heatmap by Industry × Attack Vector")
        st.markdown( "<h1 style='font-size: 16px; text-align: center;'>CVaR Heatmap by Industry × Attack Vector</h1>", 
            unsafe_allow_html=True )
        pivot = df.pivot_table(values='Simulated CVaR 95% ($)', index='Industry', columns='Attack Vector', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrBr", ax=ax, annot_kws={"fontsize": 8})
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

        st.pyplot(fig)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            # Create quantile-based bins for Risk Exposure Score
            df['Risk Tier'] = pd.qcut(
            df['Risk Exposure Score'], q=3, labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            # Map tiers to colors
            tier_colors = { 'High Risk': 'red', 'Medium Risk': 'gold', 'Low Risk': 'green' }
            df['Color'] = df['Risk Tier'].map(tier_colors)

            st.markdown( "<h1 style='font-size: 16px;text-align: center;'>CVaR vs Residual Risk (Bubble Size = Exposure Score)</h1>", 
            unsafe_allow_html=True )
            fig, ax = plt.subplots(figsize=(10, 9))
            ax.scatter(
                    x=df['Residual Loss ($)'],
                    y=df['Simulated CVaR 95% ($)'],
                    s=df['Risk Exposure Score'] * 1000,  # Scale bubble size
                    c=df['Color'],
                    alpha=0.8,
                    edgecolors='black',
                    linewidth=0.5
                )

            # Add quadrant lines
            x_median = df['Residual Loss ($)'].median()
            y_median = df['Simulated CVaR 95% ($)'].median()
            ax.axvline(x=x_median, color='gray', linestyle='--')
            ax.axhline(y=y_median, color='gray', linestyle='--')

            # Labels and title
            ax.set_xlabel("Residual Loss ($)")
            ax.set_ylabel("Simulated CVaR 95% ($)")

            # Create manual legend
            import matplotlib.patches as mpatches
            legend_handles = [
            mpatches.Patch(color='red', label='High Risk'),
            mpatches.Patch(color='gold', label='Medium Risk'),
            mpatches.Patch(color='green', label='Low Risk')
            ]
            ax.legend(handles=legend_handles, title="Risk Tier", bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

        with col2:
            # Initialize lists for outputs
            simulated_means = []
            simulated_cvars = []

            # Number of simulations
            n_simulations = 10000
            portfolio_annual_losses = []


            # Loop through each record
            for idx, row in df.iterrows():
                lambda_freq = row['Frequency']
                residual_loss = row['Residual Loss ($)']
                
                # If frequency is zero, assume zero expected loss
                if lambda_freq == 0 or residual_loss == 0:
                    simulated_means.append(0.0)
                    simulated_cvars.append(0.0)
                else:
                    # Estimate per-event loss for lognormal sampling
                    avg_loss_per_event = residual_loss / lambda_freq
                    mu = np.log(avg_loss_per_event + 1e-6)
                    sigma = 0.8  # can be tuned
                    
                    # Simulate frequency
                    freq_sim = np.random.poisson(lam=lambda_freq, size=n_simulations)
                    
                    # Simulate per-event loss
                    loss_per_event = np.random.lognormal(mean=mu, sigma=sigma, size=n_simulations)
                    
                    # Total simulated loss = frequency × severity
                    total_loss = freq_sim * loss_per_event
                    
                    # Store statistics
                    simulated_means.append(np.mean(total_loss))
                    tail_losses = total_loss[total_loss >= np.percentile(total_loss, 95)]
                    simulated_cvars.append(np.mean(tail_losses))

            # Add results to DataFrame
            df['Simulated Mean Loss ($)'] = simulated_means
            df['Simulated CVaR 95% ($)'] = simulated_cvars
            # For each simulation, sample from the simulated mean losses of all records
            for _ in range(n_simulations):
                # Random number of events per year: based on total expected frequency
                total_events = np.random.poisson(lam=df['Frequency'].sum())
                
                if total_events == 0:
                    portfolio_annual_losses.append(0.0)
                else:
                    # Randomly sample 'total_events' losses from the Simulated Mean Losses
                    sampled_losses = np.random.choice(df['Simulated Mean Loss ($)'].values, size=total_events, replace=True)
                    portfolio_annual_losses.append(np.sum(sampled_losses))

            # Convert to numpy array for further analysis
            portfolio_annual_losses = np.array(portfolio_annual_losses)

            # Preview: display summary statistics
            #import matplotlib.pyplot as plt
            st.markdown( "<h1 style='font-size: 16px;text-align: center;'>Simulated Portfolio Annual Loss Distribution</h1>", 
            unsafe_allow_html=True )
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.hist(portfolio_annual_losses, bins=50, color='skyblue', edgecolor='black', alpha=0.8)
            ax.axvline(np.mean(portfolio_annual_losses), color='red', linestyle='dashed', label='Mean Loss')
            ax.axvline(np.percentile(portfolio_annual_losses, 95), color='darkorange', linestyle='dashed', label='95% VaR')
            ax.axvline(np.mean(portfolio_annual_losses[portfolio_annual_losses > np.percentile(portfolio_annual_losses, 95)]), 
                        color='purple', linestyle='dashed', label='95% CVaR')
            ax.set_xlabel("Total Annual Loss ($)")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)



                
    
if __name__ == "__main__":
    main()