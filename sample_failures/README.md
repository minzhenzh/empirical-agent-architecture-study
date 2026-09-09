# Failed-run samples

Examples for the brief trace inspection in Section 4.2.2. Each directory
contains the agent trajectory and evaluation record.

## Repeated unsuccessful actions

`workbench-qwen3.5-9b-0063` repeats an unsuccessful email search until the
step budget is exhausted:

> `action: email_search_emails(query=Task Update on Improve UX of feedback submission, date_min=2023-11-23, date_max=2023-11-29)`
>
> `observation: No emails found.`
>
> `Final Answer: Auto-submit: step budget exhausted`

`workbench-qwen3.5-9b-0000` repeats the same plot call four times and uses
November 30 instead of the expected November 29:

> `action: analytics_create_plot(time_min=2023-09-26, time_max=2023-11-30, value_to_plot=session_duration_seconds, plot_type=bar)`
>
> `n_expected_calls: 1`
>
> `n_predicted_calls: 4`

## Recovery and replanning

`plancraft-qwen3.5-397b-0014` crafts a stone button, clears the grid, and
replans, but does not produce the comparator within 10 steps:

> `I see that moving stone to C1 produced a stone_button in slot [0].`
>
> `I need to clear the crafting grid and set up the correct recipe for comparator.`
>
> `num_steps: 10`
>
> `success: 0`

`workbench-deepseek-v4-pro-0011` treats an empty query result as a zero-second
average and creates a plot when no action was expected:

> `observation: '{}'`
>
> `n_expected_calls: 0`
>
> `n_predicted_calls: 7`

## Execution until the limit

`plancraft-deepseek-v4-pro-0022` spends all 10 steps arranging ingredients
without producing the target:

> `action: move(slot_from=[0], slot_to=[I1], quantity=1)`
>
> `num_steps: 10`
>
> `success: 0`

`workbench-deepseek-v4-flash-0069` keeps searching for the recipient and
never creates the event or sends the required email:

> `action: directory_find_email_address(name=Sam)`
>
> `action: directory_find_email_address(name=samuel)`
>
> `Final Answer: Auto-submit: step budget exhausted`
>
> `is_correct: 0`
