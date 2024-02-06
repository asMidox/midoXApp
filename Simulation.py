import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
from scipy.stats import norm
import json
import requests
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
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

############################################################################################
# Page d'accueil
def home_page():   
    
    with st.expander("click"):
       st.write("""
        Bienvenue sur notre interface de simulation financière, un projet développé dans le cadre de notre cours de calcul stochastique. 
        Cette application vous offre la possibilité d'explorer et de comprendre le comportement des marchés financiers à travers des simulations interactives.

        Notre objectif est de vous permettre d'évaluer divers scénarios financiers en utilisant des modèles stochastiques, tels que le Mouvement Brownien Standard et la Simulation de Monte Carlo.

        Pour commencer, choisissez l'une des simulations dans le menu de gauche. Chaque simulation propose des paramètres spécifiques que vous pouvez ajuster selon vos préférences.

        Profitez de votre expérience de simulation et n'hésitez pas à explorer les différentes facettes de la finance stochastique!
        """)
       

    #def load_lottiefile(filepath: str):
    #    with open(filepath, "r") as f:
    #       return json.load(f)

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    #lottie_gif= load_lottiefile(r"C:\Users\user\Downloads\gif.json")
    lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")
    #st_lottie(
    #    lottie_gif
    #)

    st_lottie(
        lottie_hello
    )
    st.title("Bienvenue sur l'interface de simulation financière")


    
    # Introduction au projet
    

    # Instructions pour choisir une simulation
    st.subheader("Comment commencer?")
    st.write("""
    - **Mouvement Brownien Standard :** Explorez la trajectoire aléatoire d'un actif financier avec des variations aléatoires.
    - **Mouvement Brownien Géométrique :** Simulez le prix d'un actif financier en tenant compte de la volatilité et du taux de dérive.
    - **Simulation de Monte Carlo :** Obtenez des perspectives sur le prix futur d'une action en utilisant des simulations stochastiques.

    Choisissez une simulation dans le menu de gauche et découvrez comment les modèles stochastiques peuvent être utilisés pour mieux comprendre et anticiper les mouvements du marché financier.
    """)




############################################################################################
# Page de simulation du Mouvement Brownien Standard
def brownian_motion_page():
    st.title("Simulation du Mouvement Brownien Standard")

        # Paramètres de la simulation
    duration = st.slider("Durée de la simulation (jours)", 1, 365, 30)
    volatility = st.slider("Volatilité (écart-type)", 0.1, 3.0, 0.5)

        # Action pour lancer la simulation
    if st.button("Lancer la simulation"):
            
            total3, total4,  = st.columns(2)

            with total4:
                st.info('Durée ', icon="⌛")
                st.metric(label="Durée de la simulation", value=f"{duration} jrs")

            with total3:
                st.info('Volatilité', icon="💣")
                st.metric(label="Volatilité", value=f"{volatility:.2f}")

            # Style pour les cartes métriques
            style_metric_cards(background_color="#FFFFFF", border_left_color="#686664", border_color="#000000",box_shadow="#F71938")

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




#######################################################################################
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
        total1, total2, total3, total4, total5 = st.columns(5)

        with total1:
            st.info('prix intial', icon="💰")
            st.metric(label="Prix initial", value=f"{initial_price:,.2f}")

        with total2:
            st.info('dérive', icon="🔰")
            st.metric(label="Taux de dérive", value=f"{drift_rate:.2f}")

        with total3:
            st.info('Volatilité', icon="💣")
            st.metric(label="Volatilité", value=f"{volatility:.2f}")

        with total4:
            st.info('Durée ', icon="⌛")
            st.metric(label="Durée de la simulation", value=f"{duration} jrs")

        with total5:
            st.info('Nombre', icon="🔢")
            st.metric(label="Nombre de simulations", value=f"{num_simulations}")

        # Style pour les cartes métriques
        style_metric_cards(background_color="#FFFFFF", border_left_color="#686664", border_color="#000000",box_shadow="#F71938")

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



###########################################################################################
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
        total1, total2, total5 = st.columns(3)

        with total1:
            st.info('ticker', icon="💰")
            st.metric(label="ticker", value=f"{ticker:}")

        with total2:
            st.info('Début', icon="🔰")
            st.metric(label="Début", value=f"{start_date:}")

        with total5:
            st.info('Nombre', icon="🔢")
            st.metric(label="Nombre de simulations", value=f"{num_simulations}")

        # Style pour les cartes métriques
        style_metric_cards(background_color="#FFFFFF", border_left_color="#686664", border_color="#000000",box_shadow="#F71938")


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




#############################################################################################
# Fonction pour calculer le prix de l'option européenne selon la formule de Monte Carlo
def monte_carlo_option_price(S, K, T, r, sigma, option_type='call', num_simulations=10000):
    dt = T / 252
    num_steps = int(T / dt)
    np.random.seed(42)

    simulated_prices = np.zeros((num_simulations, num_steps + 1))
    simulated_prices[:, 0] = S

    for i in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_simulations)
        simulated_prices[:, i] = simulated_prices[:, i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    if option_type == 'call':
        payoff = np.maximum(simulated_prices[:, -1] - K, 0)
    elif option_type == 'put':
        payoff = np.maximum(K - simulated_prices[:, -1], 0)
    else:
        payoff = None

    option_price = np.exp(-r * T) * np.mean(payoff)

    return option_price

# Fonction pour simuler une trajectoire de prix
def simulate_price_trajectory(S, r, sigma, T, num_simulations):
    dt = T / 252
    num_steps = int(T / dt)
    np.random.seed(42)
    simulated_prices = np.zeros((num_simulations, num_steps + 1))
    simulated_prices[:, 0] = S

    for i in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_simulations)
        simulated_prices[:, i] = simulated_prices[:, i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # Création du graphe des simulations avec Plotly Express
    fig = px.line(simulated_prices.T, labels={"value": "Stock Price", "index": "Steps"}, title="Simulated Price Trajectories")
    
    # Supprimer la légende pour éviter l'affichage d'un grand nombre de simulations
    fig.update_layout(showlegend=False)

    # Affichage du graphe dans Streamlit
    st.plotly_chart(fig)

    df_simulated_prices = pd.DataFrame(simulated_prices, columns=[f"Step {i}" for i in range(num_steps + 1)])
    return df_simulated_prices

# Page de simulation des options européennes
def european_options_page():
    st.title("Simulation des Options Européennes")
    with st.expander("click"):
       st.write("""
        Les options européennes sont des contrats d'options qui peuvent être exercés uniquement à la date d'expiration. 
        La méthode de Monte Carlo est une approche de simulation stochastique qui peut être utilisée pour estimer le prix 
        d'une option en prenant en compte le comportement aléatoire du prix de l'actif sous-jacent.
        """)


    # Paramètres de l'option
    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        option_type = st.radio("Type d'option", ["Call", "Put"], key="option_type")
        underlying_price = st.number_input("Prix actuel de l'actif sous-jacent (S)", value=100.0, step=1.0, key="underlying_price")
        strike_price = st.number_input("Prix d'exercice de l'option (K)", value=100.0, step=1.0, key="strike_price")
        
    with col2:
        interest_rate = st.number_input("Taux d'intérêt annuel (r)", value=0.05, step=0.01, key="interest_rate")
        volatility = st.number_input("Volatilité annuelle (σ)", value=0.2, step=0.01, key="volatility")
        num_simulations = st.number_input("Nombre de simulations", value=10000, step=1000, key="num_simulations")


    # Nombre de simulations
    time_to_expiry = st.slider("Durée jusqu'à l'expiration (en jours)", 1, 365, 30)
    #st.slider("Durée de la simulation (jours)", 1, 365, 30)

    # Action pour calculer le prix de l'option par Monte Carlo
    if st.button("Calculer le prix de l'option par Monte Carlo"):
        total1, total2,total3,total4= st.columns(4)
        
        with total1:
            st.info("Type d'option", icon="💰")
            st.metric(label="Type d'option", value=f"{option_type:}")

        with total2:
            st.info('Prix actuel', icon="🔰")
            st.metric(label="S", value=f"{underlying_price:}")

        with total3:
            st.info("Prix d'exercice ", icon="🔢")
            st.metric(label="K", value=f"{strike_price}")

        with total4:
            st.info("Taux d'intérêt ", icon="💰")
            st.metric(label="r", value=f"{interest_rate:}")
        
        total5,total6,total7 = st.columns(3)
        
        with total5:
            st.info('Volatilité annuelle', icon="🔰")
            st.metric(label="σ", value=f"{volatility:}")
        
        with total6:
            st.info("Echéance", icon="🔢")
            st.metric(label="T", value=f"{time_to_expiry:}jrs")

        with total7:
            st.info('Nombre de simulations', icon="🔢")
            st.metric(label="Nombre de simulations", value=f"{num_simulations}")
        
        # Style pour les cartes métriques
        style_metric_cards(background_color="#FFFFFF", border_left_color="#686664", border_color="#000000",box_shadow="#F71938")


        # Calcul du prix de l'option avec Monte Carlo
        option_price = monte_carlo_option_price(underlying_price, strike_price, time_to_expiry, interest_rate, volatility, option_type.lower(), num_simulations)

        # Affichage du résultat
        st.subheader("Résultat de la simulation Monte Carlo")
        st.write(f"Prix de l'option {option_type} : {option_price:.4f}")

        # Ajout d'un exemple de trajectoire de prix simulée avec Plotly Express
        st.subheader("Exemple de trajectoire de prix simulée")
        simulate_price_trajectory(underlying_price, interest_rate, volatility, time_to_expiry, num_simulations)






# Sélecteur de page
#page_selector = st.sidebar.radio("Sélectionnez une simulation", ["Accueil", "Mouvement Brownien Standard", "Mouvement Brownien Géométrique", "Monte Carlo", "Options Européennes"])

def sideBar():
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Accueil", "Mouvement Brownien Standard", "Mouvement Brownien Géométrique", "Monte Carlo", "Options Européennes"],
            icons=["house","","","",""],
            menu_icon="cast",
            default_index=0
        )
    return selected

page_selector = sideBar()

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

