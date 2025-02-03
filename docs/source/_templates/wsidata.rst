{{ fullname | escape | underline }}


.. currentmodule:: {{ module }}


{% if objtype in ['class'] %}
.. auto{{ objtype }}:: {{ objname }}
    :special-members: __call__

{% else %}
.. auto{{ objtype }}:: {{ objname }}

{% endif %}
