# CMake generated Testfile for 
# Source directory: /home/d3dx13/catkin_ws/src/cv_camera
# Build directory: /home/d3dx13/catkin_ws/src/cv_camera/cmake-build-debug
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(_ctest_cv_camera_rostest_test_cv_camera.test "/home/d3dx13/catkin_ws/src/cv_camera/cmake-build-debug/catkin_generated/env_cached.sh" "/usr/bin/python2" "/opt/ros/melodic/share/catkin/cmake/test/run_tests.py" "/home/d3dx13/catkin_ws/src/cv_camera/cmake-build-debug/test_results/cv_camera/rostest-test_cv_camera.xml" "--return-code" "/usr/bin/python2 /opt/ros/melodic/share/rostest/cmake/../../../bin/rostest --pkgdir=/home/d3dx13/catkin_ws/src/cv_camera --package=cv_camera --results-filename test_cv_camera.xml --results-base-dir \"/home/d3dx13/catkin_ws/src/cv_camera/cmake-build-debug/test_results\" /home/d3dx13/catkin_ws/src/cv_camera/test/cv_camera.test ")
set_tests_properties(_ctest_cv_camera_rostest_test_cv_camera.test PROPERTIES  _BACKTRACE_TRIPLES "/opt/ros/melodic/share/catkin/cmake/test/tests.cmake;160;add_test;/opt/ros/melodic/share/rostest/cmake/rostest-extras.cmake;52;catkin_run_tests_target;/opt/ros/melodic/share/rostest/cmake/rostest-extras.cmake;80;add_rostest;/opt/ros/melodic/share/rostest/cmake/rostest-extras.cmake;100;_add_rostest_google_test;/home/d3dx13/catkin_ws/src/cv_camera/CMakeLists.txt;90;add_rostest_gtest;/home/d3dx13/catkin_ws/src/cv_camera/CMakeLists.txt;0;")
add_test(_ctest_cv_camera_rostest_test_no_yaml.test "/home/d3dx13/catkin_ws/src/cv_camera/cmake-build-debug/catkin_generated/env_cached.sh" "/usr/bin/python2" "/opt/ros/melodic/share/catkin/cmake/test/run_tests.py" "/home/d3dx13/catkin_ws/src/cv_camera/cmake-build-debug/test_results/cv_camera/rostest-test_no_yaml.xml" "--return-code" "/usr/bin/python2 /opt/ros/melodic/share/rostest/cmake/../../../bin/rostest --pkgdir=/home/d3dx13/catkin_ws/src/cv_camera --package=cv_camera --results-filename test_no_yaml.xml --results-base-dir \"/home/d3dx13/catkin_ws/src/cv_camera/cmake-build-debug/test_results\" /home/d3dx13/catkin_ws/src/cv_camera/test/no_yaml.test ")
set_tests_properties(_ctest_cv_camera_rostest_test_no_yaml.test PROPERTIES  _BACKTRACE_TRIPLES "/opt/ros/melodic/share/catkin/cmake/test/tests.cmake;160;add_test;/opt/ros/melodic/share/rostest/cmake/rostest-extras.cmake;52;catkin_run_tests_target;/opt/ros/melodic/share/rostest/cmake/rostest-extras.cmake;80;add_rostest;/opt/ros/melodic/share/rostest/cmake/rostest-extras.cmake;100;_add_rostest_google_test;/home/d3dx13/catkin_ws/src/cv_camera/CMakeLists.txt;94;add_rostest_gtest;/home/d3dx13/catkin_ws/src/cv_camera/CMakeLists.txt;0;")
subdirs("gtest")
