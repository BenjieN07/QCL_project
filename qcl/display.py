"""Native Python figure window; plotting files remain usable without a desktop."""


def show_figure(figure, title):
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    try:
        window = tk.Tk()
    except tk.TclError:
        print('A desktop window is unavailable. Open the saved PNG to inspect the results.')
        return
    window.title(title)
    window.geometry('1100x760')
    window.minsize(700, 500)
    canvas = FigureCanvasTkAgg(figure, master=window)
    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas.draw()
    window.mainloop()
