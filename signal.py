import numpy as np
import plotly.graph_objects as go

class Signal:
    def __init__(self, E: np.ndarray, I: np.ndarray):
        self.E = E
        self.I = I

    def pplot(E: np.ndarray, I: np.ndarray, start_idx: int=0, end_idx: int=-1, peak_data: dict = None) -> None:
        '''
        Preety plot. Plots the signal using plotly.
        It shows the peak signal (max value) and the left and the right margin of the active signal.
        
        Arguments:
            E (np.ndarray): the potential signal in µV
            I (np.ndarray): the current singnal in µA
            start_idx (int): starting index for peak search
            end_idx (int): ending index for peak search
            plotting (bool): whether to plot the peak point on the signal (default is False)
        Returns:
            None
        '''
        fig = go.Figure()
        if not peak_data:
            fig.add_trace(go.Scatter(x=E[start_idx:end_idx], y=I[start_idx:end_idx], mode="lines", name="signal"))
        else:
            # plotting the signal
            fig.add_trace(go.Scatter(x=E[start_idx:end_idx], y=I[start_idx:end_idx], mode="lines", name="signal"))
            # plotting the area
            fig.add_trace(go.Scatter(x=E[peak_data['idx_left']:peak_data['idx_right'] + 1], y=I[peak_data['idx_left']:peak_data['idx_right'] + 1], mode="lines", name="signal", fill="tozeroy"))
            # plotting starting of the peak
            fig.add_trace(go.Scatter(x=[peak_data['E_left']], y=[peak_data['I_left']], mode="lines+markers", name="left"))
            # plotting ending of the peak
            fig.add_trace(go.Scatter(x=[peak_data['E_right']], y=[peak_data['I_right']], mode="lines+markers", name="right"))
            # plotting the peak
            fig.add_trace(go.Scatter(x=[peak_data['Ep']], y=[peak_data['Ip']], mode="lines+markers", name="peak"))
            
        fig.update_layout(
            # width=750,
            height=750,
            template="plotly_white"
        )
        fig.update_xaxes(
            showgrid=True,
            minor=dict(showgrid=True)
        )
        fig.update_yaxes(
            showgrid=True,
            minor=dict(showgrid=True)
        )
        fig.show()

    def normalie():
        pass
    def get_peak_value():
        pass
    def get_peak_potential():
        pass
    def get_peak_value():
        pass
