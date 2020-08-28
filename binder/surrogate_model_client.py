import copy
from bokeh.io import output_notebook, show
from bokeh.layouts import column, row, Spacer
from bokeh.models import Div
from bokeh.models.widgets import Select
from bokeh import palettes

from bokeh.io import curdoc
#from bokeh.resources import INLINE#
from lume_model.utils import load_variables
from lume_epics.client.widgets.plots import ImagePlot, Striptool
from lume_epics.client.widgets.controls import build_sliders
from lume_epics.client.widgets.tables import ValueTable
from lume_epics.client.controller import Controller



prefix = "test" # Prefix used by our server
variable_filename =  "binder/files/surrogate_model_variables.pickle"
input_variables, output_variables = load_variables(variable_filename)

controller = Controller("ca")

inputs = [input_variables['maxb(2)'], 
    input_variables['phi(1)'], 
    input_variables['total_charge:value'], 
    #input_variables['in_xmin'], input_variables['in_xmax'], 
    #input_variables['in_ymin'], input_variables['in_ymax']
]

sliders = build_sliders(inputs, controller, prefix)

# create image plot
image_plot = ImagePlot([output_variables["x:y"]], controller, prefix)

# build the plot using specific bokeh palette
pal = palettes.viridis(256)
image_plot.build_plot(palette=pal)


output_variables_to_display = copy.deepcopy(output_variables)
output_variables_to_display.pop("x:y")
output_variables_to_display = list(output_variables_to_display.values())

# Set up the striptool
striptool = Striptool(output_variables_to_display, controller, "test")
striptool.build_plot()

# set up global pv
current_striptool_pv = striptool.live_variable

# set up selection
def striptool_select_callback(attr, old, new):
    global current_striptool_pv
    current_striptool_pv = new


striptool_select = Select(
    title="Variable to plot:",
    value=current_striptool_pv,
    options=list(striptool.pv_monitors.keys()),
)

striptool_select.on_change("value", striptool_select_callback)

# striptool data update callback
def striptool_update_callback():
    """
    Calls striptool update with the current global process variable.
    """
    global current_striptool_pv
    striptool.update(live_variable = current_striptool_pv)


# add table
value_table = ValueTable(output_variables_to_display, controller, "test")

# Set up table update callback
def table_update_callback():
    """
    Updates the value table.
    """
    value_table.update()


striptool.plot.height = 450
striptool.plot.width = 450
image_plot.plot.width = 510
image_plot.plot.height = 475
value_table.table.height= 450



# render
curdoc().title = "LCLS Sc Injector"

#row(Div(text="LCLS Cu Injector")),
bokeh_sliders = [slider.bokeh_slider for slider in sliders]
curdoc().add_root(
            column(
            row(Spacer(width = 540), Div(text="<h1 style='text-align:center;'>LCLS Cu Injector</h1>")),

            row(
                    column(striptool_select, striptool.plot), 
                    column(Spacer(height=10), value_table.table), column(Spacer( height=10), image_plot.plot),
            ),
            row(
                Spacer(width=350),column(bokeh_sliders[:4]), column(bokeh_sliders[4:])
                ) ,
            )
    )

curdoc().add_periodic_callback(image_plot.update, 250)
curdoc().add_periodic_callback(table_update_callback, 250)
curdoc().add_periodic_callback(striptool_update_callback, 250)
for slider in sliders:
    curdoc().add_periodic_callback(slider.update, 250)