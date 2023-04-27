import tkinter as tk


def get_input(prompt1, prompt2, dropdown_options, radio_options, default_radio_option):
    # Create a new tkinter window
    window = tk.Tk()

    # Create a label to display the first prompt
    label1 = tk.Label(window, text=prompt1)
    label1.pack()

    # Create a variable to store the selected option
    selected_option = tk.StringVar(window)

    # Create a dropdown box for the user to select the first value
    dropdown = tk.OptionMenu(window, selected_option, *dropdown_options)
    dropdown.pack()

    # Create a label to display the second prompt
    label2 = tk.Label(window, text=prompt2)
    label2.pack()

    # Create a variable to store the selected option
    selected_option2 = tk.StringVar(window)

    # Create a radio button for each option
    for option in radio_options:
        radio_button = tk.Radiobutton(window, text=option, variable=selected_option2, value=option)
        radio_button.pack()

    # Set the default value for the second radio button
    selected_option2.set(default_radio_option)

    # Define a function to get the input from the dropdown box and radio buttons
    def submit_input():
        value1 = selected_option.get()
        value2 = selected_option2.get()
        window.destroy()
        return (value1, value2)

    # Create a button to submit the input
    button = tk.Button(window, text="Submit", command=submit_input)
    button.pack()

    # Start the tkinter event loop
    window.mainloop()
    # Return None if the window was closed without submitting the input
    if not selected_option.get() or not selected_option2.get():
        return None
    return (selected_option.get(), selected_option2.get())


def print_output(output):
    # Create a new tkinter window
    window = tk.Tk()

    # Create a text widget to display output
    text = tk.Text(window, height=10, width=50)
    text.pack()

    # Insert the output into the text widget
    text.insert(tk.END, output)

    # Start the tkinter event loop
    window.mainloop()
