#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LaTeX Table Generator

This module provides functions for generating formatted LaTeX tables with color-coded cells.
The tables can be directly used in LaTeX documents with the colortbl package.

Acknowledgments:
    This code is based on the work from the ASVspoof 2021 Challenge:
    - ASVspoof 2021 Challenge: https://www.asvspoof.org/
    - Original authors: ASVspoof 2021 Challenge organizers and contributors

Copyright (c) 2021 ASVspoof Challenge
Copyright (c) 2024 Audio Deepfake Evaluation Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import matplotlib.pyplot as plt

def return_valid_number_idx(data_array):
    """Return indices of valid numbers in the array."""
    return np.where(~np.isnan(data_array))[0]

def return_latex_color_cell(x, value_min, value_max, colorscale, colorwrap, color_func):
    """Return LaTeX color command for a cell."""
    if np.isnan(x):
        return ""
    
    if value_min == value_max:
        normalized_x = 0.5
    else:
        normalized_x = (x - value_min) / (value_max - value_min)
    
    if colorwrap > 0:
        normalized_x = 1 - np.exp(-colorwrap * normalized_x)
    
    color = color_func(normalized_x * colorscale)
    return f"\\cellcolor[rgb]{{{color[0]:.3f},{color[1]:.3f},{color[2]:.3f}}}"

def fill_cell(content, width, sep=""):
    """Fill a cell with content and padding."""
    return f"{content:{width}}{sep}"

def return_one_row_latex(row):
    """Return a row in LaTeX format."""
    return " & ".join(row) + " \\\\\n"

def return_one_row_text(row):
    """Return a row in text format."""
    return "".join(row) + "\n"

def print_table(data_array, column_tag, row_tag, n_synthesizers,
                data_array2 = None,
                print_format = "1.2f", 
                with_color_cell = True,
                colormap='Greys', 
                colorscale = 0.5, 
                colorwrap = 0, 
                col_sep = '', 
                print_latex_table=True, 
                print_text_table=True,
                print_format_along_row=True,
                color_minmax_in = 'global',
                pad_data_column = 0,
                pad_dummy_col = 0):
    """
    print a latex table given the data (np.array) and tags    
    step1. table will be normalized so that values will be (0, 1.0)
    step2. each normalzied_table[i,j] will be assigned a RGB color tuple 
           based on color_func( normalzied_table[i,j] * color_scale)
    input
    -----
      data_array: np.array [M, N]
      column_tag: list of str, length N, tag in the first row
      row_tag: list of str, length M, tags in first col of each row
      
      print_format: str or list of str, specify the format to print number
                    default "1.2f"
      print_format_along_row: bool, when print_format is a list, is this
                    list specified for rows? Default True
                    If True, row[n] will use print_format[n]
                    If False, col[n] will use print_format[n]
      with_color_cell: bool, default True,
                      whether to use color in each latex cell
      colormap: str, color map name (matplotlib)
      colorscale: float, default 0.5, 
                    normalized table value will be scaled 
                    color = color_func(nomrlized_table[i,j] * colorscale)
                  list of float
                    depends on configuration of color_minmax_in
                    if color_minmax_in = 'row', colorscale[i] for the i-th row
                    if color_minmax_in = 'col', colorscale[j] for the j-th row
                  np.array
                    color_minmax_in cannot be 'row' or 'col'. 
                    colorscale[i, j] is used for normalized_table[i, j]
      colorwrap: float, default 0, wrap the color-value mapping curve
                 colorwrap > 0 works like mels-scale curve
      col_sep: str, additional string to separate columns. 
               You may use '\t' or ',' for CSV
      print_latex_table: bool, print the table as latex command (default True)
      print_text_table: bool, print the table as text format (default True)
      color_minmax_in: how to decide the max and min to compute cell color?
                 'global': get the max and min values from the input matrix 
                 'row': get the max and min values from the current row
                 'col': get the max and min values from the current column
                  (min, max): given the min and max values
                 default is global
      pad_data_column: int, pad columns on the left or right of data matrix
                  (the tag column will still be on the left)
                  0: no padding (default)
                  -N: pad N dummy data columns to the left
                   N: pad N dummy data columns to the right
      pad_dummy_col: int, pad columns to the left or right of the table
                  (the column will be padded to the left of head column)
                  0: no padding (default)
                  N: pad N columns to the left
    output
    ------
      latext_table, text_table
      
    Tables will be printed to the screen.
    The latex table will be surrounded by begin{tabular}...end{tabular}
    It can be directly pasted to latex file.
    However, it requires usepackage{colortbl} to show color in table cell.    
    """
    if column_tag is None:
        column_tag = ["" for data in data_array[0, :]]
    if row_tag is None:
        row_tag = ["" for data in data_array]

    if pad_data_column < 0:
        column_tag = ["" for x in range(-pad_data_column)] + column_tag
        dummy_col = np.zeros([data_array.shape[0], -pad_data_column]) + np.nan
        data_array = np.concatenate([dummy_col, data_array], axis=1)
    elif pad_data_column > 0:
        column_tag = ["" for x in range(pad_data_column)] + column_tag
        dummy_col = np.zeros([data_array.shape[0], pad_data_column]) + np.nan
        data_array = np.concatenate([data_array, dummy_col], axis=1)
    else:
        pass

    # check print_format
    if type(print_format) is not list:
        if print_format_along_row:
            # repeat the tag
            print_format = [print_format for x in row_tag]
        else:
            print_format = [print_format for x in column_tag]
    else:
        if print_format_along_row:
            assert len(print_format) == len(row_tag)
        else:
            assert len(print_format) == len(column_tag)

    # color configuration
    color_func = plt.cm.get_cmap(colormap)
    
    def get_latex_color(data_array, row_idx, col_idx, color_minmax_in):
        x = data_array[row_idx, col_idx]
        if color_minmax_in == 'row':
            data_idx = return_valid_number_idx(data_array[row_idx])
            value_min = np.min(data_array[row_idx][data_idx])
            value_max = np.max(data_array[row_idx][data_idx])
            if type(colorscale) is list:
                colorscale_tmp = colorscale[row_idx]
        elif color_minmax_in == 'col':
            data_idx = return_valid_number_idx(data_array[:, col_idx])
            value_min = np.min(data_array[:, col_idx][data_idx])
            value_max = np.max(data_array[:, col_idx][data_idx])    
            if type(colorscale) is list:
                colorscale_tmp = colorscale[col_idx]
        elif type(color_minmax_in) is tuple or type(color_minmax_in) is list:
            value_min = color_minmax_in[0]
            value_max = color_minmax_in[1]
            if type(colorscale) is np.ndarray:
                colorscale_tmp = colorscale[row_idx, col_idx]
        else:
            data_idx = return_valid_number_idx(data_array)
            value_min = np.min(data_array[data_idx])
            value_max = np.max(data_array[data_idx])
            if type(colorscale) is np.ndarray:
                colorscale_tmp = colorscale[row_idx, col_idx]
            
        if type(colorscale) is not list:
            colorscale_tmp = colorscale
            
        # return a color command for latex cell
        return return_latex_color_cell(x, value_min, value_max, 
                                       colorscale_tmp, colorwrap, color_func)
    
    # maximum width for tags in 1st column
    row_tag_max_len = max([len(x) for x in row_tag])

    # maximum width for data and tags for other columns
    if print_format_along_row:
        tmp_len = []
        for idx, data_row in enumerate(data_array):
            tmp_len.append(
                max([len("{num:{form}}".format(num=x, form=print_format[idx])) \
                     for x in data_row]))
    else:
        tmp_len = []
        for idx, data_col in enumerate(data_array.T):
            tmp_len.append(
                max([len("{num:{form}}".format(num=x, form=print_format[idx])) \
                     for x in data_col]))
    col_tag_max_len = max([len(x) for x in column_tag] + tmp_len)
    
    # prepare buffer
    text_buffer = ""
    latex_buffer = ""
    text_cell_buffer = []
    latex_cell_buffer = []

    # latex head
    if pad_dummy_col > 0:
        latex_buffer += r"\begin{tabular}{" \
                        + ''.join(['c' for x in column_tag + ['']])
        latex_buffer += ''.join(['c' for x in range(pad_dummy_col)]) + r"}"+"\n"
    else:
        latex_buffer += r"\begin{tabular}{" \
                        + ''.join(['c' for x in column_tag + ['']]) + r"}"+"\n"

    latex_buffer += r"\toprule" + "\n"
    
    # head row
    #  for latex
    hrow = [fill_cell("Method", row_tag_max_len)] \
           + [fill_cell(x, col_tag_max_len) for x in column_tag]
    if pad_dummy_col > 0:
        hrow = [fill_cell("", 1) for x in range(pad_dummy_col)] + hrow

    latex_buffer += return_one_row_latex(hrow)
    latex_cell_buffer.append(hrow)

    latex_buffer += r"\midrule \multicolumn{3}{l}{\textit{max. EER of $M=" + str(n_synthesizers) + r"$ synthesizers}}\\" + "\n"

    #  for plain text (add additional separator for each column)
    hrow = [fill_cell("Method", row_tag_max_len, col_sep)] \
           + [fill_cell(x, col_tag_max_len, col_sep) for x in column_tag]
    text_buffer += return_one_row_text(hrow)
    text_cell_buffer.append(hrow)

    # contents
    row = data_array.shape[0]
    col = data_array.shape[1]
    
    for i in range(row):
        # for latex
        if with_color_cell:
            row_latex = [fill_cell(row_tag[i], row_tag_max_len)]
            for j in range(col):
                color_cmd = get_latex_color(data_array, i, j, color_minmax_in)
                if print_format_along_row:
                    fmt = print_format[i]
                else:
                    fmt = print_format[j]
                if np.isnan(data_array[i, j]):
                    cell = fill_cell("", col_tag_max_len)
                else:
                    cell = fill_cell("{num:{form}}".format(num=data_array[i, j], form=fmt), col_tag_max_len)
                row_latex.append(color_cmd + cell)
        else:
            row_latex = [fill_cell(row_tag[i], row_tag_max_len)]
            for j in range(col):
                if print_format_along_row:
                    fmt = print_format[i]
                else:
                    fmt = print_format[j]
                if np.isnan(data_array[i, j]):
                    cell = fill_cell("", col_tag_max_len)
                else:
                    cell = fill_cell("{num:{form}}".format(num=data_array[i, j], form=fmt), col_tag_max_len)
                row_latex.append(cell)
        
        if pad_dummy_col > 0:
            row_latex = [fill_cell("", 1) for x in range(pad_dummy_col)] + row_latex
        
        latex_buffer += return_one_row_latex(row_latex)
        latex_cell_buffer.append(row_latex)

        # for plain text
        row_text = [fill_cell(row_tag[i], row_tag_max_len, col_sep)]
        for j in range(col):
            if print_format_along_row:
                fmt = print_format[i]
            else:
                fmt = print_format[j]
            if np.isnan(data_array[i, j]):
                cell = fill_cell("", col_tag_max_len, col_sep)
            else:
                cell = fill_cell("{num:{form}}".format(num=data_array[i, j], form=fmt), col_tag_max_len, col_sep)
            row_text.append(cell)
        
        text_buffer += return_one_row_text(row_text)
        text_cell_buffer.append(row_text)

    # Print average data rows if provided
    if data_array2 is not None:
        latex_buffer += r"\midrule \multicolumn{3}{l}{\textit{avg. EER of $M=" + str(n_synthesizers) + r"$ synthesizers}}\\" + "\n"
        
        avg_row = data_array2.shape[0]
        avg_col = data_array2.shape[1]
        
        for i in range(avg_row):
            # for latex
            if with_color_cell:
                row_latex = [fill_cell(f"Avg-{row_tag[i]}", row_tag_max_len)]
                for j in range(avg_col):
                    color_cmd = get_latex_color(data_array2, i, j, color_minmax_in)
                    if print_format_along_row:
                        fmt = print_format[i]
                    else:
                        fmt = print_format[j]
                    if np.isnan(data_array2[i, j]):
                        cell = fill_cell("", col_tag_max_len)
                    else:
                        cell = fill_cell("{num:{form}}".format(num=data_array2[i, j], form=fmt), col_tag_max_len)
                    row_latex.append(color_cmd + cell)
            else:
                row_latex = [fill_cell(f"Avg-{row_tag[i]}", row_tag_max_len)]
                for j in range(avg_col):
                    if print_format_along_row:
                        fmt = print_format[i]
                    else:
                        fmt = print_format[j]
                    if np.isnan(data_array2[i, j]):
                        cell = fill_cell("", col_tag_max_len)
                    else:
                        cell = fill_cell("{num:{form}}".format(num=data_array2[i, j], form=fmt), col_tag_max_len)
                    row_latex.append(cell)
            
            if pad_dummy_col > 0:
                row_latex = [fill_cell("", 1) for x in range(pad_dummy_col)] + row_latex
            
            latex_buffer += return_one_row_latex(row_latex)
            latex_cell_buffer.append(row_latex)

            # for plain text
            row_text = [fill_cell(f"Avg-{row_tag[i]}", row_tag_max_len, col_sep)]
            for j in range(avg_col):
                if print_format_along_row:
                    fmt = print_format[i]
                else:
                    fmt = print_format[j]
                if np.isnan(data_array2[i, j]):
                    cell = fill_cell("", col_tag_max_len, col_sep)
                else:
                    cell = fill_cell("{num:{form}}".format(num=data_array2[i, j], form=fmt), col_tag_max_len, col_sep)
                row_text.append(cell)
            
            text_buffer += return_one_row_text(row_text)
            text_cell_buffer.append(row_text)

    latex_buffer += r"\bottomrule" + "\n"
    latex_buffer += r"\end{tabular}" + "\n"

    if print_latex_table:
        print(latex_buffer)
    if print_text_table:
        print(text_buffer)

    return latex_buffer, text_buffer 