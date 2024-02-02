import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
from scipy.stats import norm

APP_ICON_URL = "https://bit.ly/42viGwA"
st.set_page_config(page_icon=APP_ICON_URL)


# Fonction pour le style des composants
def apply_style():
    st.markdown(
        """
        <style>
            .st-eb {
                background-color: #f0f0f0;
            }

            .st-bi {
                background-color: #4682B4;
                color: white;
            }

            .css-1t42fv2 {
                color: #4682B4;
            }

            .css-1v3fvcr {
                color: #4682B4;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Custom CSS for styling
custom_css = """
    <style>
        body {
            background-color: rgb(129, 164, 182);
            color: #FFFFFF;
        }
        [data-testid=stSidebar] {
            background-color: rgb(129, 164, 182);
            color: #FFFFFF;
        }
        [aria-selected="true"] {
            color: #000000;
        }
       
    </style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Page d'accueil
def home_page():   
    st.title("Bienvenue sur l'interface de simulation financière")
    st.write("Sélectionnez une simulation dans le menu de gauche.")


# Page de simulation du Mouvement Brownien Standard
def brownian_motion_page():
    st.title("Simulation du Mouvement Brownien Standard")

        # Paramètres de la simulation
    duration = st.slider("Durée de la simulation (jours)", 1, 365, 30)
    volatility = st.slider("Volatilité (écart-type)", 0.1, 3.0, 0.5)

        # Action pour lancer la simulation
    if st.button("Lancer la simulation"):
            # Génération du Mouvement Brownien Standard
            dt = 1  # Pas de temps
            num_steps = int(duration / dt)
            t = np.linspace(0, duration, num_steps)
            increments = np.random.normal(0, np.sqrt(dt) * volatility, num_steps)
            brownian_path = np.cumsum(increments)

            df = pd.DataFrame({"Temps (jours)": t, "Valeur générée": brownian_path})
            # Tracé du graphe avec Plotly
            fig = px.line(df, x="Temps (jours)", y="Valeur générée", labels={"Valeur générée": "Valeur"})
            fig.update_layout(
                title="Simulation du Mouvement Brownien Standard",
                xaxis_title="Temps (jours)",
                yaxis_title="Valeur",
                legend_title="Trajectoire du Mouvement Brownien"
            )
            st.plotly_chart(fig)


            # Appercu des valeurs générées
            st.subheader("Aperçu des valeurs générées:")
            st.write("Durée de la simulation:", duration, "jours")
            st.write("Volatilité (écart-type):", volatility)
            st.write(df)

# Page de simulation du Mouvement Brownien Géométrique
def geometric_brownian_motion_page():
    st.title("Simulation du Mouvement Brownien Géométrique")

    # Paramètres de la simulation
    initial_price = st.number_input("Prix initial", value=100.0, step=1.0)
    drift_rate = st.number_input("Taux de dérive (drift rate)", value=0.05, step=0.01)
    volatility = st.number_input("Volatilité (écart-type)", value=0.2, step=0.01)
    duration = st.slider("Durée de la simulation (jours)", 1, 365, 30)
    num_simulations = st.number_input("Nombre de simulations", value=1, step=1)

    # Action pour lancer la simulation
    if st.button("Lancer la simulation"):
        # Génération des Mouvements Browniens Géométriques
        dt = 1  # Pas de temps
        num_steps = int(duration / dt)
        t = np.linspace(0, duration, num_steps)

        # Générer plusieurs simulations
        fig = px.line()
        all_simulated_values = []  # Pour stocker toutes les valeurs générées
        for i in range(num_simulations):
            increments = np.random.normal((drift_rate - 0.5 * volatility**2) * dt, volatility * np.sqrt(dt), num_steps)
            geometric_brownian_path = initial_price * np.exp(np.cumsum(increments))

            # Ajouter la simulation à la figure
            fig.add_trace(go.Scatter(x=t, y=geometric_brownian_path, mode='lines', name=f"Simulation {i + 1}"))

            # Stocker toutes les valeurs générées
            all_simulated_values.append(geometric_brownian_path)

        # Afficher un aperçu sur toutes les valeurs générées
        st.subheader("Aperçu sur toutes les valeurs générées:")
        all_simulated_df = pd.DataFrame(all_simulated_values).transpose()
        all_simulated_df.columns = [f"Simulation {i + 1}" for i in range(num_simulations)]
        st.dataframe(all_simulated_df)

        # Mise en forme de la figure
        fig.update_layout(
            title="Simulation du Mouvement Brownien Géométrique",
            xaxis_title="Temps (jours)",
            yaxis_title="Valeur",
            legend_title="Trajectoire du Mouvement Brownien Géométrique"
        )

        st.plotly_chart(fig)


# Page de simulation de Monte Carlo
def monte_carlo_page():
    apply_style()
    st.title("Simulation de Monte Carlo pour le Prix d'une Action")

    # Saisie du ticker de l'action
    ticker = st.text_input("Entrez le ticker de l'action (par exemple, AAPL)", "AAPL")

    # Période de simulation
    start_date = st.date_input("Date de début de simulation", pd.to_datetime('2022-01-01'))

    # Nombre de simulations
    num_simulations = st.number_input("Nombre de simulations", value=1, step=1)

    # Bouton pour lancer la simulation
    if st.button("Lancer la simulation", key="simulate_button"):
        # Récupération des données historiques de l'action
        try:
            stock_data = yf.download(ticker, start=start_date)
        except:
            st.warning("Erreur : Impossible de récupérer les données historiques. Vérifiez le ticker et les dates sélectionnées.")
            return

        # Afficher les données historiques
        st.subheader("Données historiques de l'action:")
        st.dataframe(stock_data.style.set_properties(**{'background-color': '#f0f0f0', 'color': 'black'}), use_container_width=True)

        # Prix initial de l'action
        initial_price = stock_data['Adj Close'].iloc[-1]
        st.subheader("Informations sur l'action:")
        st.write(f"Le prix initial du stock est : {initial_price:.2f}")

        # Date d'aujourd'hui
        today_date = datetime.today().strftime('%Y-%m-%d')
        st.write(f"La date d'aujourd'hui est: {today_date}")

        # Volatilité du stock
        returns = np.log(1 + stock_data['Adj Close'].pct_change())
        volatility = returns.std()
        st.write(f"La volatilité du stock est : {volatility:.4f}")

        # Simulation de Monte Carlo
        all_sims = []
        mu = returns.mean()

        for i in range(num_simulations):
            sim_rets = np.random.normal(mu, volatility, len(stock_data))
            sim_prices = initial_price * (sim_rets + 1).cumprod()
            all_sims.append(sim_prices)

        # Dataframe avec les simulations
        df_info = pd.DataFrame(all_sims)
        df_info_transposed = df_info.transpose()

        st.subheader("Informations sur les simulations:")
        st.dataframe(df_info_transposed.style.set_properties(**{'background-color': '#f0f0f0', 'color': 'black'}), use_container_width=True)

        # Tracé des simulations
        st.subheader("Graphique des simulations:")
        fig = px.line(df_info_transposed, title="Simulation de Monte Carlo pour le Prix d'une Action")
        fig.update_layout(xaxis_title="Jours", yaxis_title="Prix simulé")
        st.plotly_chart(fig)



# Fonction pour calculer le prix de l'option européenne selon la formule de Black-Scholes
def black_scholes_option_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        option_price = None

    return option_price

# Page de simulation des options européennes
def european_options_page():
    st.title("Simulation des Options Européennes")

    # Paramètres de l'option
    st.subheader("Paramètres de l'option")
    option_type = st.radio("Type d'option", ["Call", "Put"])
    st.write(f"Type d'option sélectionné : {option_type}")

    underlying_price = st.number_input("Prix actuel de l'actif sous-jacent (S)", value=100.0, step=1.0)
    strike_price = st.number_input("Prix d'exercice de l'option (K)", value=100.0, step=1.0)
    time_to_expiry = st.number_input("Durée jusqu'à l'expiration (en années)", value=1.0, step=0.1)
    interest_rate = st.number_input("Taux d'intérêt annuel (r)", value=0.05, step=0.01)
    volatility = st.number_input("Volatilité annuelle (σ)", value=0.2, step=0.01)

    # Action pour calculer le prix de l'option
    if st.button("Calculer le prix de l'option"):
        # Calcul du prix de l'option
        option_price = black_scholes_option_price(underlying_price, strike_price, time_to_expiry, interest_rate, volatility, option_type.lower())

        # Affichage du résultat
        st.subheader("Résultat de la simulation")
        st.write(f"Prix de l'option {option_type} : {option_price:.4f}")

# Sélecteur de page
page_selector = st.sidebar.radio("Sélectionnez une simulation", ["Accueil", "Mouvement Brownien Standard", "Mouvement Brownien Géométrique", "Monte Carlo", "Options Européennes"])

# Affichage de la page sélectionnée
if page_selector == "Accueil":
    home_page()
elif page_selector == "Mouvement Brownien Standard":
    brownian_motion_page()
elif page_selector == "Mouvement Brownien Géométrique":
    geometric_brownian_motion_page()
elif page_selector == "Monte Carlo":
    monte_carlo_page()
elif page_selector == "Options Européennes":
    european_options_page()
