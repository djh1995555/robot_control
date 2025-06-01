#!/usr/bin/env python
import os
from typing import OrderedDict
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from src.utils.env import ROBOT_CONTROL_ROOT_DIR
from src.utils.visualizer.report_plotter import ReportPlotter


class ReportGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        config_filepath = os.path.join(ROBOT_CONTROL_ROOT_DIR, 'config', self.cfg['task'], 'report_generator_cfg.yaml')
        with open(config_filepath, 'r') as f:
            self.target_signals = yaml.load(f, Loader=yaml.FullLoader)
        self.report_plotter = ReportPlotter('ReportGenerator')
        self.data_selected = OrderedDict()
        self.figure_height = 600

    def generate_report(self, data_logger):
        subplot_figure = None
        plot_html_str = ""  

        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        output_dir = os.path.join(ROBOT_CONTROL_ROOT_DIR, 'output', self.cfg['task'], formatted_time)
        if(not os.path.exists(output_dir)):
            os.makedirs(output_dir)
        output_filename = os.path.join(output_dir, f"{self.cfg['task']}.html")

        all_data = data_logger.get_data()
        target_panel = dict(self.target_signals["target_panel"])
        for sub_panel_name, signal_names in target_panel.items():
            sub_dict = OrderedDict()
            for signal_name in signal_names:
                if signal_name in all_data.keys():
                    data = all_data[signal_name]
                    time = all_data['timestamp']
                    sub_dict[signal_name] = (data, time)
            self.data_selected[sub_panel_name] = sub_dict


        start_timestamp = all_data['timestamp'][0]
        figure_list = []
        for sub_panel_name, sub_dict in self.data_selected.items():
            legend_list = []
            value_list = []
            time_list = []
            for signal_full_name, (data, data_time) in sub_dict.items():
                time_list.append(np.array([(x - start_timestamp) for x in data_time]))
                value_list.append(np.array(data))
                legend_list.append(signal_full_name)
            
            subplot = self.report_plotter.plot_figure_plotly(x_list = time_list, 
                                                y_list = value_list,
                                                legend_list = legend_list,
                                                x_label = 'time / s',
                                                y_label = '',
                                                title = sub_panel_name,
                                                legend_prefix = '',
                                                figure_height=self.figure_height,)
            figure_list.append(subplot)  

        subplot_figure_list = [(i + 1, 1, fig) for i, fig in enumerate(figure_list)]
        subplot_figure = self.report_plotter.append_figure_to_subplot_plotly(
            subplot_figure_list, 
            len(figure_list), 
            1, 
            template="plotly_dark", 
            subplot_fig=subplot_figure, 
            vertical_spacing=0.05
        )
        plot_html_str += self.report_plotter.get_fuel_fig_html_str({"Comparison": subplot_figure})
        html_str = self.report_plotter.generate_html_fuel_report(plot_html_str)
        with open(output_filename, 'w') as f:
            f.write(html_str)