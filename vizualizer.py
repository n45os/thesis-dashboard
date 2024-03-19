import base64
import hashlib
import json
import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.helpers import get_color
from analysis.loaders import (
    count_adversarial_data,
    count_benign_data,
    get_existing_adversaries_fractions,
    load_data,
    only_adversarial_data,
    only_benign_data,
)

st.set_page_config(
    layout="wide",  # wide, centered
    initial_sidebar_state="auto",
)


def create_benign_trace(round_data, config_data):
    if "accuracy" in round_data:
        accuracy_df = pd.DataFrame(
            round_data["accuracy"], columns=["Round", "Accuracy"]
        )
        color = get_color(config_data)
        trace = go.Scatter(
            x=accuracy_df["Round"],
            y=accuracy_df["Accuracy"],
            mode="lines",
            line=dict(dash="dot", width=4),  # Thicker dotted line
            marker=dict(color=color),
            name=f'benign {config_data.get("strategy", {}).get("name", "Unknown")}',
        )
        return trace
    else:
        return None


def filter_and_sort_advers_data(data_list, filter_options):
    filtered_data = []
    for data in data_list:
        # if adversaries_fraction is 0, then it is benign
        if data[1].get("adversaries_fraction", 0) == 0:
            continue
        config = data[1]
        if (
            filter_options["strategy_enabled"]
            and config.get("strategy", {}).get("name")
            not in filter_options["strategy_name"]
        ):
            continue
        if (
            filter_options["advers_fn_enabled"]
            and config.get("adversaries", {}).get("advers_fn")
            not in filter_options["advers_fn"]
        ):
            continue
        if (
            filter_options["dataset_enabled"]
            and config.get("dataset", {}).get("name")
            not in filter_options["dataset_name"]
        ):
            continue
        if filter_options["num_clients_enabled"] and not (
            filter_options["num_clients_min"]
            <= config.get("num_clients", 0)
            <= filter_options["num_clients_max"]
        ):
            continue
        # Exclude 0 adversary fraction if not enabled
        if (
            filter_options["adversary_fraction_enabled"]
            and config.get("adversaries_fraction", 0)
            not in filter_options["adversary_fraction_list"]
        ):
            continue

        filtered_data.append(data)
    return filtered_data


def filter_and_sort_benign_data(data_list, filter_options):
    filtered_data = []
    for data in data_list:
        config = data[1]
        if (
            filter_options["strategy_enabled"]
            and config.get("strategy", {}).get("name")
            not in filter_options["strategy_name"]
        ):
            continue
        if (
            filter_options["dataset_enabled"]
            and config.get("dataset", {}).get("name")
            not in filter_options["dataset_name"]
        ):
            continue
        if filter_options["num_clients_enabled"] and not (
            filter_options["num_clients_min"]
            <= config.get("num_clients", 0)
            <= filter_options["num_clients_max"]
        ):
            continue
        filtered_data.append(data)
    return filtered_data


def display_config(config_data):
    att = config_data.get("adversaries", {}).get("advers_fn", "N/A")
    config_str = (
        f" ⚙️ Model: **{config_data.get('dataset', {}).get('name', 'N/A')}** with **{config_data.get('strategy', {}).get('name', 'N/A')}** | "
        f"👥 **{config_data.get('num_clients', 'N/A')}** clients \n\n "
        f"💀 **{config_data.get('adversaries_fraction', 'N/A') * 100}%** with **{att}**"
    )
    if att == "minmax" or att == "minsum" or att == "tailored":
        config_str += f" | **{config_data.get('adversaries', {}).get('perturbation_type', 'N/A')}**"
    # if fraction is 0, then it is benign
    if config_data.get("adversaries_fraction", 0) == 0:
        config_str = (
            f"⚙️ Model: {config_data.get('dataset', {}).get('name', 'N/A')} with {config_data.get('strategy', {}).get('name', 'N/A')} | "
            f"👥 {config_data.get('num_clients', 'N/A')} clients \n\n "
            f"😇 Benign "
        )
    st.write(f"{config_str}")


def display_timing(timing_data):
    #       "timing": {
    #     "start": "2024-02-17 14:13:17",
    #     "end": "2024-02-17 14:34:45",
    #     "elapsed_time": 1287.9041640758514
    #   }
    # format elapsed time to h:mm:ss
    formatted_elapsed_time = str(timing_data.get("elapsed_time", "N/A"))
    if formatted_elapsed_time != "N/A":
        formatted_elapsed_time = str(
            pd.to_datetime(float(formatted_elapsed_time), unit="s").strftime("%H:%M:%S")
        )

    formated_end_time = str(timing_data.get("end", "N/A"))
    if formated_end_time != "N/A":
        # readable format Mon 24 Feb, 2024 14:34:45
        formated_end_time = str(
            pd.to_datetime(formated_end_time).strftime("%a %d %b, %Y %H:%M:%S")
        )

    timing_str = (
        f"🕒 Ended {formated_end_time} | " f"⌛️ Elapsed Time: {formatted_elapsed_time} "
    )
    st.write(f"{timing_str}")


@st.cache_resource
def create_visualization(round_data, config_data):
    fig = None
    if "accuracy" in round_data:
        accuracy_df = pd.DataFrame(
            round_data["accuracy"], columns=["Round", "Accuracy"]
        )
        fig = px.line(
            accuracy_df,
            x="Round",
            y="Accuracy",
            title="Accuracy Over Rounds",
            markers=True,
            line_shape="spline",
            render_mode="svg",
        )
        color = get_color(config_data)
        for trace in fig.data:
            trace.line.color = color
        fig.update_layout(
            xaxis_title="Round",
            yaxis_title="Accuracy",
            # set only upper range to 1 and lower range to auto
            yaxis=dict(range=[-0.15, 1]),
            legend_title="Metric",
        )
        last_point = accuracy_df.iloc[-1]
        fig.add_annotation(
            x=last_point["Round"],
            y=last_point["Accuracy"],
            text=f"Final accuracy: {last_point['Accuracy']:.2f}",
            showarrow=True,
            arrowhead=1,
            font=dict(
                size=14,  # Increase text size
            ),
            arrowcolor="red",  # Change arrow color
        )
        if "impact" or "round_impact" in round_data:
            if "round_impact" in round_data:
                impact_df = pd.DataFrame(
                    round_data["round_impact"], columns=["Round", "Impact"]
                )
                fig.add_bar(
                    x=impact_df["Round"],
                    y=impact_df["Impact"],
                    name="Impact",
                    marker_color="red",
                    opacity=0.7,
                )
            elif "impact" in round_data:
                impact_df = pd.DataFrame(
                    round_data["impact"], columns=["Round", "Impact"]
                )
                fig.add_bar(
                    x=impact_df["Round"],
                    y=impact_df["Impact"],
                    name="Impact",
                    marker_color="red",
                    opacity=0.7,
                )
        else:
            fig.update_layout(title="Benign Data (No Impact)")
    else:
        st.warning("Missing accuracy data in the selected file.")

    # hash the round data to create a unique identifier for this graph
    graph_id = hashlib.md5(json.dumps(round_data).encode("utf-8")).hexdigest()
    graph_id = "graph-" + graph_id[:8]

    # New: Unique identifier for the checkbox
    checkbox_id = "checkbox-" + graph_id

    return fig, graph_id, checkbox_id


@st.cache_resource
def combine_graphs(selected_graphs, filtered_data, filtered_benign_data=None):
    combined_fig = make_subplots()

    # Predefined set of colors for distinction
    colors = px.colors.qualitative.Plotly

    if filtered_benign_data is not None:
        # benign_data = retrieve_benign_data_for_config(config_data, filtered_data)
        for round_data, config_data, _ in only_benign_data(filtered_benign_data):
            benign_trace = create_benign_trace(round_data, config_data)
            if benign_trace:
                combined_fig.add_trace(benign_trace)

    # Iterate through each selected graph
    for i, graph_id in enumerate(selected_graphs):
        # Find the corresponding figure in filtered_data
        for round_data, config_data, _ in filtered_data:
            current_id = hashlib.md5(json.dumps(round_data).encode("utf-8")).hexdigest()
            current_id = "graph-" + current_id[:8]
            if current_id == graph_id:
                fig, _, _ = create_visualization(round_data, config_data)
                legend_group_name = f"group_{i}"
                for trace in fig["data"]:
                    # Clone the trace to modify it
                    new_trace = trace
                    # Update trace name with strategy and adversaries fraction
                    strategy_name = config_data.get("strategy", {}).get(
                        "name", "Unknown"
                    )
                    advers_fraction = config_data.get("adversaries_fraction", 0) * 100
                    advers_fn = config_data.get("adversaries", {}).get(
                        "advers_fn", "Unknown"
                    )
                    new_trace.name = (
                        f"{strategy_name} ({advers_fraction}%) w/ {advers_fn}"
                    )
                    new_trace.legendgroup = legend_group_name
                    color = get_color(config_data)
                    if "marker" in new_trace:  # For bar charts, etc.
                        new_trace.marker.color = color
                    if "line" in new_trace:  # For line charts
                        new_trace.line.color = color
                    # For impact graphs, set showlegend to False
                    if "Impact" in new_trace.name:
                        new_trace.showlegend = False
                    # Add the modified trace to the combined figure
                    combined_fig.add_trace(new_trace)

    combined_fig.update_layout(
        title="Combined Graphs",
        xaxis_title="Round",
        yaxis_title="Accuracy",
        legend_title="Metrics",
        # yaxis=dict(range=[0, 0.7]),
    )

    return combined_fig


st.title("Data Visualization Dashboard for Federated Learning Model Poisoning Attacks")


with st.expander("**Information and Instructions**"):
    st.write(
        "This is an interactive dashboard for visualizing the results of federated learning simulations in the presence of adversaries. "
        "The dashboard allows you to filter and compare the results of different simulations based on various parameters such as the dataset, the number of clients, the adversary fraction, and the aggregation strategy. "
        "\n\n"
        "**Results presented in this work have been produced using the Aristotle University "
        "of Thessaloniki (AUTh) High Performance Computing Infrastructure and Resources.**"
    )

    st.markdown("""
    ### Instructions
- **Apply Filters:** Refine your visualization by using filters such as adversarial strategy, number of clients, and adversary fractions.
- **Explore Visualizations:** Interactive charts and graphs display the simulation results in the main panel. Hover over chart elements for more details or adjust your filters to observe the effects of different parameters on the outcomes.
- **Combine and Compare Results:** Select multiple adversarial simulations to combine them into a single graph for comparison. Use the checkboxes to select the graphs you want to combine, then click the 'Combine Selected Graphs' button to generate the combined graph.
    """)

folder_path = "./res_complete"
data_list = load_data(folder_path)

existing_adversaries_fractions = get_existing_adversaries_fractions(data_list)

st.sidebar.write("# Filters")

advers_fn_enabled = st.sidebar.checkbox(
    "Filter by Adversarial Function",
    help="Select weather or not to filter by Adversarial Function",
    value=True,
)
advers_fn = st.sidebar.multiselect(
    "Adversarial Function",
    ["minmax", "minsum", "tailored", "lie", "pga"],
    # default=["lie", "pga", "minmax", "minsum", "tailored"],
    default=["lie"],
)

st.sidebar.write("---")

dataset_enabled = st.sidebar.checkbox(
    "Filter by Dataset", help="Select weather or not to filter by Dataset", value=True
)
dataset_name = st.sidebar.multiselect(
    "Dataset Name",
    ["mnist", "femnist", "cifar10", "cifar100"],
    # default=["femnist", "cifar100"]
    default=["femnist"],
)

st.sidebar.write("---")

strategy_enabled = st.sidebar.checkbox(
    "Filter by Strategy",
    help="Select weather or not to filter by the aggregation method",
    value=False,
)

strategy_name = st.sidebar.multiselect(
    "Strategy Name",
    [
        "fed_avg",
        "multikrum",
        "krum",
        "fed_median",
        "bulyan",
        "trimmed_mean",
        "dnc",
        "cc",
    ],
    # default=["dnc", "fed_avg", "multikrum", "krum", "fed_median", "bulyan", "trimmed_mean", "cc"],
    default=["multikrum"],
)

st.sidebar.write("---")

adversary_fraction_enabled = st.sidebar.checkbox(
    "Filter by Adversary Fraction",
    help="Select weather or not to filter by Adversary Fraction",
    value=True,
)
# checkboxes for selecting the adversary fractions based on the existing ones. returns a list of selected fractions
adversary_fractions_list = st.sidebar.multiselect(
    # in percentage
    "Adversary Fraction",
    list(existing_adversaries_fractions),
    default=list(existing_adversaries_fractions)[1],
)

st.sidebar.write("---")

num_clients_enabled = st.sidebar.checkbox(
    "Filter by Number of Clients",
    help="Select weather or not to filter by Number of Clients",
    value=True,
)
num_clients_range = st.sidebar.slider("Number of Clients", 0, 150, (50, 120))


filter_options = {
    "strategy_enabled": strategy_enabled,
    "strategy_name": strategy_name,
    "advers_fn_enabled": advers_fn_enabled,
    "advers_fn": advers_fn,
    "dataset_enabled": dataset_enabled,
    "dataset_name": dataset_name,
    "num_clients_enabled": num_clients_enabled,
    "num_clients_min": num_clients_range[0],
    "num_clients_max": num_clients_range[1],
    "adversary_fraction_enabled": adversary_fraction_enabled,
    "adversary_fraction_list": adversary_fractions_list,
}

filtered_data = filter_and_sort_advers_data(data_list, filter_options)
filtered_benign_data = filter_and_sort_benign_data(data_list, filter_options)

benign_filtered_data = only_benign_data(filtered_benign_data)
adversarial_filtered_data = only_adversarial_data(filtered_data)

total_results_len = len(data_list)
total_benign_results_len = len(filtered_benign_data)
filtered_results_len = len(filtered_data)
filtered_benign_results_len = len(benign_filtered_data)


st.write(
    f":mag: Filtered **{filtered_results_len}/{count_adversarial_data(data_list=data_list)}** adversary {'result' if filtered_results_len==1 else 'results'} & **{filtered_benign_results_len}/{count_benign_data(data_list=data_list)}** benign {'result' if filtered_benign_results_len==1 else 'results'} out of **{total_results_len}** total results."
)
st.markdown("---")


col1, spacer, col2 = st.columns([1, 0.2, 1])

with col1:
    # Initialize the session state for the button if it doesn't already exist
    if "all_graphs_selected" not in st.session_state:
        st.session_state.all_graphs_selected = False

    # Define a function to toggle the button state
    def toggle_all_graphs_selection():
        st.session_state.all_graphs_selected = not st.session_state.all_graphs_selected

    # create_strategy_comparison_graph(filtered_data)
    selected_graphs = []

    st.markdown("## :innocent: Benign Data")
    st.write(
        f"""
            *This section contains {len(benign_filtered_data)} {'chart' if len(benign_filtered_data)==1 else 'charts'} of **benign data** based on the filters selected.
            The benign data is used as a baseline for comparison with the adversarial data and includes simulations with no adversaries.*
            """
    )
    with st.expander(
        f"**View {len(benign_filtered_data)} {'chart' if len(benign_filtered_data)==1 else 'charts'}**",
        expanded=False,
    ):  # Set expanded=True if you want it to be expanded by default
        for round_data, config_data, timing_data in benign_filtered_data:
            display_config(config_data)
            fig, graph_id, checkbox_id = create_visualization(round_data, config_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")

    # Display individual graphs
    st.markdown("## :skull: Adversarial Data")
    st.write(
        f"""
            *This section contains {len(adversarial_filtered_data)} {'chart' if len(adversarial_filtered_data)==1 else 'charts'} of **adversarial data** based on the filters selected.
            The adversarial data includes simulations with adversaries and is used to compare the impact of adversaries on the model.*
            """
    )
    # Create a button that toggles the state
    st.button(
        "**Select all graphs**",
        on_click=toggle_all_graphs_selection,
        type="primary",
        help="Select all graphs to combine them into a single graph in the Combined Graphs section.",
    )

    with st.expander(
        f"**View {len(adversarial_filtered_data)} {'chart' if len(adversarial_filtered_data)==1 else 'charts'}**",
        expanded=False,
    ):
        # st.markdown("---")
        for round_data, config_data, timing_data in adversarial_filtered_data:
            display_config(config_data)
            fig, graph_id, checkbox_id = create_visualization(round_data, config_data)
        
            if fig:
                # Display the checkbox with each graph
                if st.session_state.all_graphs_selected:
                    # When all graphs are selected, individual toggles reflect this selection
                    toggle = st.toggle(
                        "Combine this graph",
                        key=checkbox_id,
                        value=True,  # Reflects global selection
                        help="This graph is already selected. Deselect the 'Select all graphs' option to combine graphs individually.",
                    )
                else:
                    # Individual selection toggles
                    toggle = st.toggle(
                        "Combine this graph",
                        key=checkbox_id,
                        value=st.session_state.get(
                            checkbox_id, False
                        ),  # Use existing state, default to False
                        help="Select to combine this graph individually.",
                    )
                if toggle:
                    selected_graphs.append(graph_id)

                st.plotly_chart(fig, use_container_width=True)
                # st.write(fig)
            st.markdown("---")

# st.markdown("---")

with col2:
    # Use an expander for the combined graph
    st.markdown("## Combined Graphs")

    st.markdown("*Select adversial graphs from the left column and select one of the the benign options below. Then, press the button to generate the combined graph.*")

    include_benign = st.checkbox("Include All Benign Data", value=False)

    # add an opiton include best benign data
    include_best_benign = st.checkbox("Include Best Benign Data", value=True)

    if not st.session_state.all_graphs_selected:
        st.markdown("#### *Select adversial graphs to combine.*")
        st.markdown("*After selecting the graphs, click the button to generate them.*")
    combine_button = st.button(
        "**Combine Selected Graphs**",
        # help="Combine the selected graphs into a single graph.",
        type="primary",
    )


# Function to convert Plotly figure to SVG
def fig_to_svg(fig):
    return fig.to_image(format="svg", width=800, height=600)


# Function to create a download button for SVG
def create_download_button(svg_bytes, filename="graph.svg", button_text="Download SVG"):
    b64 = base64.b64encode(svg_bytes).decode()
    href = f'<a href="data:image/svg+xml;base64,{b64}" download="{filename}" style="margin-top: 0.75rem;">{button_text}</a>'
    st.markdown(href, unsafe_allow_html=True)


combined_fig = None
if combine_button:
    with col2:
        if len(selected_graphs) == 0 and not include_benign:
            st.warning(
                "No graphs selected for combination. Select at least one graph or use the 'Select all graphs' toggle."
            )
        elif len(selected_graphs) == 0 and include_benign:
            combined_fig = combine_graphs(
                selected_graphs, filtered_data, filtered_benign_data
            )
            st.plotly_chart(combined_fig)
        else:
            if include_benign:
                combined_fig = combine_graphs(
                    selected_graphs, filtered_data, filtered_benign_data
                )
                st.plotly_chart(combined_fig)

            elif include_best_benign:
                # get best bening data. It will be the one that the end accuracy is the highest
                benign_best = None
                best_accuracy = 0
                for one in benign_filtered_data:
                    if "accuracy" in one[0]:
                        accuracy = one[0]["accuracy"][-1][1]
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            benign_best = one
                print(benign_best)
                combined_fig = combine_graphs(
                    selected_graphs, filtered_data, [benign_best]
                )

                st.plotly_chart(combined_fig)
            else:
                combined_fig = combine_graphs(
                    selected_graphs, adversarial_filtered_data
                )
                st.plotly_chart(combined_fig)
