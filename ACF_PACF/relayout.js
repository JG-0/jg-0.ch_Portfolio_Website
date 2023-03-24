window.handleRelayout = function (eventData) {
    const plotDiv = document.getElementById(eventData.plotId);
    const trace1Visible = plotDiv.data[0].visible;
    const trace2Visible = plotDiv.data[1].visible;

    if (trace1Visible && trace2Visible) {
        Plotly.react(plotDiv, plotDiv.data, { ...plotDiv.layout, autosize: true });
    } else {
        Plotly.react(plotDiv, plotDiv.data, { ...plotDiv.layout, autosize: false });
    }
};
