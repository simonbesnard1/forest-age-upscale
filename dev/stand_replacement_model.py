import matplotlib.pyplot as plt
import numpy as np

# Implementing independent stand replacement events

# Creating separate curves for each stand replacement event
# These curves start with gradual aging and then have a stand replacement event at specified ages
stand_replacement_curves = []

extended_forest_age = np.arange(0, 201, 1)
replacement_impacts = [0.1, 0, 0.3]  # Different impacts for each event

# Adjusting the gradual aging curve for the extended age range
extended_gradual_aging = 1 - np.exp(-extended_forest_age / 50)

# Implementing multiple stand replacement events at specified ages
stand_replacement_ages = [25, 60, 150]
for age, impact in zip(stand_replacement_ages, replacement_impacts):
    # Copy the gradual aging curve
    curve = np.copy(extended_gradual_aging)

    # Apply the impact of the stand replacement event at the specified age
    post_event_age = extended_forest_age >= age
    curve[post_event_age] = impact * curve[age - 1]

    # Simulate recovery after the event
    recovery = (1 - np.exp(-(extended_forest_age[post_event_age] - age) / 50)) * (1 - impact)
    curve[post_event_age] += recovery

    # Add the curve to the list
    stand_replacement_curves.append(curve)

fig, ax = plt.subplots(1, 1, figsize=(7.3, 5), gridspec_kw={'wspace': 0, 'hspace': 0.0}, constrained_layout=True)

# Plotting the curves
for i, curve in enumerate(stand_replacement_curves):
    line, = ax.plot(extended_forest_age, curve * 120, label=f"Stand replacement at {stand_replacement_ages[i]} years", linewidth=2)
    # Annotate 'Fast out' at the point of stand replacement
    #ax.axvline(x=stand_replacement_ages[i], color='red', linestyle='--', linewidth=1)
    
# Plotting gradually ageing curve
grad_line, = ax.plot(extended_forest_age, extended_gradual_aging*120, label="Gradually ageing", color='black',  linewidth=3.5)

# Annotate 'Slow in' somewhere on the gradually ageing curve
slow_in_age = 100  # Example age, choose appropriately
ax.annotate('Slow in', xy=(slow_in_age, extended_gradual_aging[slow_in_age]*120), 
            xytext=(20, 30), textcoords='offset points', ha='center', 
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3'), 
            fontsize=16, color='black', fontweight= 'bold')

# Scenario 1: Gradual aging by ten years
start_age = 50
end_age = start_age + 10
start_carbon = extended_gradual_aging[start_age] * 120
end_carbon = extended_gradual_aging[end_age] * 120

ax.scatter([start_age, end_age], [start_carbon, end_carbon], color='blue', s=200)
#ax.arrow(start_age, start_carbon, end_age - start_age, end_carbon - start_carbon, head_width=3, head_length=3, fc='blue', ec='blue')

ax.set_xlabel('Forest age [years]', fontsize=16)
ax.set_ylabel('Carbon stock [MgC ha$^{-1}$]', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right']. set_visible(False)
ax.tick_params(labelsize=14)
plt.legend(frameon=False, fontsize=12)
ax.annotate('t0', xy=(start_age, start_carbon), 
            xytext=(0, 10), textcoords='offset points',
            ha='center', va='bottom', fontsize=16, color='blue')

# Annotate the year 2020 on the second blue dot
ax.annotate('t1', xy=(end_age, end_carbon), 
            xytext=(0, 10), textcoords='offset points',
            ha='center', va='bottom', fontsize=16, color='blue')
ax.set_title('Gradually ageing vs. stand-replaced forests', fontsize=18, fontweight='bold', pad=20)
plt.savefig('/home/simon/Documents/science/GFZ/presentation/EGU2024_talk_sbesnard/images/standReplacement_age_a.png', dpi=300)


fig, ax = plt.subplots(1, 1, figsize=(7.3, 5), gridspec_kw={'wspace': 0, 'hspace': 0.0}, constrained_layout=True)

# Plotting the curves
for i, curve in enumerate(stand_replacement_curves):
    line, = ax.plot(extended_forest_age, curve * 120, label=f"Stand replacement at {stand_replacement_ages[i]} years", linewidth=2)
    # Annotate 'Fast out' at the point of stand replacement
    #ax.axvline(x=stand_replacement_ages[i], color='red', linestyle='--', linewidth=1)
    if i ==2:
        ax.text(stand_replacement_ages[i], 55, 'Fast out', rotation=90, va='bottom', ha='right', fontweight= 'bold', fontsize=16, color='black')

# Plotting gradually ageing curve
grad_line, = ax.plot(extended_forest_age, extended_gradual_aging*120, label="Gradually ageing", color='black', linewidth=3.5)

# Annotate 'Slow in' somewhere on the gradually ageing curve
slow_in_age = 100  # Example age, choose appropriately

# # Scenario 2: Stand replacement at an age of 100
replacement_age = 149
pre_replacement_carbon = extended_gradual_aging[replacement_age] * 120
post_replacement_carbon = stand_replacement_curves[2][replacement_age+10] * 120

ax.scatter([replacement_age, replacement_age+10], [pre_replacement_carbon, post_replacement_carbon], color='red', s=200)
# ax.text(replacement_age, post_replacement_carbon, 'Fast out', rotation=90, va='bottom', ha='right', fontsize=16, color='red')

ax.set_xlabel('Forest age [years]', fontsize=16)
ax.set_ylabel('Carbon stock [MgC ha$^{-1}$]', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right']. set_visible(False)
ax.tick_params(labelsize=14)
plt.legend(frameon=False, fontsize=12)
ax.set_title('Gradually ageing vs. stand-replaced forests', fontsize=18, fontweight='bold', pad=20)
ax.annotate('t0', xy=(replacement_age, pre_replacement_carbon), 
            xytext=(0, 10), textcoords='offset points',
            ha='center', va='bottom', fontsize=16, color='red')

# Annotate the year 2020 on the second blue dot
ax.annotate('t1', xy=(replacement_age+10, post_replacement_carbon), 
            xytext=(0, 10), textcoords='offset points',
            ha='center', va='bottom', fontsize=16, color='red')
ax.set_title('Gradually ageing vs. stand-replaced forests', fontsize=18, fontweight='bold', pad=20)
plt.savefig('/home/simon/Documents/science/GFZ/presentation/EGU2024_talk_sbesnard/images/standReplacement_age_b.png', dpi=300)
