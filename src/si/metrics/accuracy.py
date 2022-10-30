def accuracy(y_true: list, y_pred: list) -> float:
  """Calculate the accuracy between 2 lists.

  Args:
      y_true (list): Values of the variable.
      y_pred (list): Predicted values.

  Returns:
      float: Accuracy result of the prediction
  """
  correct_values = len([value for value in y_pred if value in y_true])
  return (correct_values / len(y_true)) * 100