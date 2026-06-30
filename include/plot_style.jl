using CairoMakie
using Colors
using LaTeXStrings

# Makie sizing convention:
# 1 inch = 96 units, 1 pt = 4/3 units
const inch = 96.0
const cm   = inch / 2.54
const pt   = 4 / 3
const golden_mean = (sqrt(5) - 1) / 2

const APS_COLORS = [
    colorant"#5E4FA2",
    colorant"#3C93B8",
    colorant"#79C9A4",
    colorant"#C2E69F",
    colorant"#F1F9AA",
    colorant"#FEEC9F",
    colorant"#FDBB6C",
    colorant"#F57949",
    colorant"#D7414E",
    colorant"#9E0142",
]

function figsize(width_cm; height_width_ratio=golden_mean)
    w = width_cm * cm
    h = w * height_width_ratio
    return (w, h)
end

function aps_theme(; width_cm=8.8, height_width_ratio=golden_mean) 

    base_font  = 9pt
    label_font = 8pt
    tick_font  = 8pt
    legend_font = 8pt

    thin = 0.35pt

    axis_theme = (
        xlabelsize = label_font,
        ylabelsize = label_font,
        titlesize = base_font,

        xlabelfont = :regular,
        ylabelfont = :regular,
        titlefont = :regular,

        xlabelpadding = 2pt,
        ylabelpadding = 2pt,

        xticklabelsize = tick_font,
        yticklabelsize = tick_font,
        xticklabelpad = 2pt,
        yticklabelpad = 2pt,

        spinewidth = thin,

        xgridvisible = false,
        ygridvisible = false,

        xticksize = 2pt,
        yticksize = 2pt,
        xtickwidth = thin,
        ytickwidth = thin,

        xminorticksvisible = false,
        yminorticksvisible = false,
        xminorticksize = 0,
        yminorticksize = 0,
        xminortickwidth = thin,
        yminortickwidth = thin,

        # Matplotlib usually does not mirror ticks unless requested.
        # Set these true if you want ticks on top/right too.
        xticksmirrored = false,
        yticksmirrored = false,

        backgroundcolor = :transparent,
    )

    line_theme = (
        cycle = Cycle([:color]),
        linewidth = 1pt,
    )

    scatter_theme = (
        cycle = Cycle([:color, :marker], covary=true),
        markersize = 2pt,
        strokewidth = thin,
    )

    legend_theme = (
        labelsize = legend_font,
        framevisible = false,
        patchsize = (10pt, 8pt),
        padding = (2pt, 2pt, 2pt, 2pt),
        margin = (0pt, 0pt, 0pt, 0pt),
        rowgap = 2pt,
        colgap = 4pt,
        nbanks = 1,
    )

    colorbar_theme = (
        spinewidth = thin,
        labelsize = label_font,
        ticklabelsize = tick_font,
        ticksize = 2pt,
        tickwidth = thin,
        ticklabelpad = 2pt,
        labelpadding = 2pt,
        minorticksvisible = false,
        size = 8pt,
    )

    pal = (
        color = APS_COLORS,
        markercolor = APS_COLORS,
        patchcolor = APS_COLORS,
        marker = [:circle, :diamond, :rect, :xcross, :utriangle, :dtriangle],
        linestyle = [nothing],
    )

    fonts = (
        regular = "CMU Serif",
        bold = "CMU Serif Bold",
        italic = "CMU Serif Italic",
        bold_italic = "CMU Serif Bold Italic",
    )

    return Theme(
        fontsize = base_font,
        fonts = fonts,

        size = figsize(width_cm; height_width_ratio),
        figure_padding = 2pt,
        backgroundcolor = :white, #:transparent,

        palette = pal,

        Axis = axis_theme,
        PolarAxis = (spinewidth = thin,),
        Lines = line_theme,
        Scatter = scatter_theme,
        Legend = legend_theme,
        Colorbar = colorbar_theme,

        colgap = 8pt,
        rowgap = 8pt,
    )
end

function set_default_plot_params!(; width_cm=8.8, height_width_ratio=golden_mean)
    CairoMakie.activate!(
        type = "png",
        px_per_unit = 300 / inch,  
    )

    set_theme!(aps_theme(; width_cm, height_width_ratio))
end

function nice_points(color::String) 
    return Dict(
        :strokecolor => color, 
        :color => get_alpha_hex(color,0.65))
end

function hex_to_rgb(value ;full=false) 
    """Convert a hex color to rgb tuple."""
    value = lstrip(value,'#')
    lv = length(value)
    step = lv ÷ 3 
    scale = 1.0 / 255.0
    if full 
        scale = 1.0
    end
    col = Tuple(scale*parse(Int,value[i+begin:i+step],base=16) for i = 0:step:lv-1)
 
    return col 
end

function hex_to_rgb(value, transmit; full=false) 
    """Convert a hex color to rgb tuple."""
    value = lstrip(value,'#')
    lv = len(value)
    step = lv ÷ 3 
    scale = 1.0 / 255.0
    if full 
        scale = 1.0
    end
    col = Tuple(scale*parse(Int,value[i+begin:i+step],base=16) for i = 0:step:lv-1)
 
    return col + (transmit,) 
end

function rgb_to_hex(value) 
    """Convert a rgb tuple to a hex color string."""
    if value[1] < 1 
        scale = 255.0
    else 
        scale = 1.0
    end
    rgb = [round(Int64,scale*k) for k in value]
    return @sprintf "#%02x%02x%02x" rgb[1] rgb[2] rgb[3]  
end

function get_alpha_hex(value,alpha) 
    """Convert a hex color to an equivalent non-transparent version."""

    #first we get the rgb
    rgb = hex_to_rgb(value)

    # apply the transparency
    target = [alpha*k + (0.999-alpha) for k in rgb] 

    return rgb_to_hex(target)
end

function get_colors()
    colors = ["#0000ff","#D7414E","#008000", "#FFB347"] #["#779ECB", "#FF6961", "#FFB347", "#B8E2C8", "#CABEE8", "#D1A677"]
    return colors
end

function get_many_colors() 

    cols = [
    "#B39EB5",  # Pastel Purple
    "#C4C3E9",  # Pastel Lavender
    "#AEC6CF",  # Pastel Teal 
    "#89CFF0",  # Pastel Cyan
    "#779ECB",  # Pastel Blue  
    "#FF6961",  # Pastel Red
    "#FF9AA2",  # Pastel Coral
    "#F4A460",  # Pastel Sandy Brown
    "#FFB347",  # Pastel Orange 
    "#D1A677",  # Pastel Brown
    "#D3B3A6",  # Pastel Taupe 
    "#CFCFC4",  # Pastel Magenta 
    "#B2B982",  # Pastel Olive
    "#77DD77",  # Pastel Green
    "#DA70D6",  # Pastel Orchid 
    "#FFB6C1",   # Pastel Light Pink 
]
  
    return cols 
end
set_default_plot_params!()
