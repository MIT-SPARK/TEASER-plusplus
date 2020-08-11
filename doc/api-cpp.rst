.. _api-cpp:

C++ API
========

Solver API
----------

``teaser/registration.h`` defines the core registration solver API which provides users with the ability to solve 3D registration problems given correspondences.

.. _registration-api:

Registration Solution
^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: teaser::RegistrationSolution
   :members:
   :private-members:
   :undoc-members:

7-DoF Solver
^^^^^^^^^^^^

.. doxygenclass:: teaser::RobustRegistrationSolver
   :members:

Translation Solver
^^^^^^^^^^^^^^^^^^

.. doxygenclass:: teaser::TLSTranslationSolver
   :members:

.. doxygenclass:: teaser::AbstractTranslationSolver
   :members:

Rotation Solver
^^^^^^^^^^^^^^^

.. doxygenclass:: teaser::GNCTLSRotationSolver
   :members:

.. doxygenclass:: teaser::FastGlobalRegistrationSolver
   :members:

.. doxygenclass:: teaser::AbstractRotationSolver
   :members:

Scale Solver
^^^^^^^^^^^^

.. doxygenclass:: teaser::TLSScaleSolver
   :members:

.. doxygenclass:: teaser::ScaleInliersSelector
   :members:

.. doxygenclass:: teaser::AbstractScaleSolver
   :members:

TLS Estimator
^^^^^^^^^^^^^

.. doxygenclass:: teaser::ScalarTLSEstimator
   :members:

.. _certifier-api:

Certifier API
-------------

``teaser/certification.h`` defines the core certification API which provides users with the ability to certify a rotation solution provided by a GNC-TLS rotation solver.

Certification Result
^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: teaser::CertificationResult
   :members:
   :private-members:
   :undoc-members:

Rotation Certifier
^^^^^^^^^^^^^^^^^^

.. doxygenclass:: teaser::DRSCertifier
   :members:

.. doxygenclass:: teaser::AbstractRotationCertifier
   :members:
