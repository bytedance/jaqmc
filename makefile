# Makefile for O
O:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:O,2,ccecp,0 --config.batch_size 4096 --config.optim.iterations 30000

# Makefile for Sc
ccecp:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:Sc,1,ccecp,0 --config.batch_size 4096 --config.optim.iterations 30000
tr2:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:Sc,1,0.21_187_tr2,0 --config.batch_size 4096 --config.optim.iterations 30000
tr3:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:Sc,1,0.21_187_tr3,0 --config.batch_size 4096 --config.optim.iterations 30000
tr4:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:Sc,1,0.21_187_tr4,0 --config.batch_size 4096 --config.optim.iterations 30000
tr5:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:Sc,1,0.21_187_tr5,0 --config.batch_size 4096 --config.optim.iterations 30000
tr6:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:Sc,1,0.21_187_tr6,0 --config.batch_size 4096 --config.optim.iterations 30000
tr7:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:Sc,1,0.21_187_tr7,0 --config.batch_size 4096 --config.optim.iterations 30000
hybrid:
	python run.py --config examples/pp/lapnet/configs/hph/X.py:Sc,1,0.21_187,hybrid,0 --config.batch_size 4096 --config.optim.iterations 30000
l2:
	python run.py --config examples/pp/lapnet/configs/ph/X.py:Sc,1,0.21_187,l2,0 --config.batch_size 4096 --config.optim.iterations 30000

# Makefile for Sc+
+ccecp:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:Sc,2,ccecp,1 --config.batch_size 4096 --config.optim.iterations 30000
+tr2:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:Sc,2,0.21_187_tr2,1 --config.batch_size 4096 --config.optim.iterations 30000
+tr3:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:Sc,2,0.21_187_tr3,1 --config.batch_size 4096 --config.optim.iterations 30000
+tr4:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:Sc,2,0.21_187_tr4,1 --config.batch_size 4096 --config.optim.iterations 30000
+tr5:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:Sc,2,0.21_187_tr5,1 --config.batch_size 4096 --config.optim.iterations 30000
+tr6:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:Sc,2,0.21_187_tr6,1 --config.batch_size 4096 --config.optim.iterations 30000
+tr7:
	python run.py --config examples/pp/lapnet/configs/ecp/X.py:Sc,2,0.21_187_tr7,1 --config.batch_size 4096 --config.optim.iterations 30000
+hybrid:
	python run.py --config examples/pp/lapnet/configs/hph/X.py:Sc,2,0.21_187,hybrid,1 --config.batch_size 4096 --config.optim.iterations 30000
+l2:
	python run.py --config examples/pp/lapnet/configs/ph/X.py:Sc,2,0.21_187,l2,1 --config.batch_size 4096 --config.optim.iterations 30000

# Makefile for ScO
O_ccecp:
	python run.py --config examples/pp/lapnet/configs/ecp/XO.py:Sc,1,ccecp --config.batch_size 4096 --config.optim.iterations 100000
O_tr2:
	python run.py --config examples/pp/lapnet/configs/ecp/XO.py:Sc,1,0.21_187_tr2 --config.batch_size 4096 --config.optim.iterations 100000
O_tr3:
	python run.py --config examples/pp/lapnet/configs/ecp/XO.py:Sc,1,0.21_187_tr3 --config.batch_size 4096 --config.optim.iterations 100000
O_tr4:
	python run.py --config examples/pp/lapnet/configs/ecp/XO.py:Sc,1,0.21_187_tr4 --config.batch_size 4096 --config.optim.iterations 100000
O_tr5:
	python run.py --config examples/pp/lapnet/configs/ecp/XO.py:Sc,1,0.21_187_tr5 --config.batch_size 4096 --config.optim.iterations 100000
O_tr6:
	python run.py --config examples/pp/lapnet/configs/ecp/XO.py:Sc,1,0.21_187_tr6 --config.batch_size 4096 --config.optim.iterations 100000
O_tr7:
	python run.py --config examples/pp/lapnet/configs/ecp/XO.py:Sc,1,0.21_187_tr7 --config.batch_size 4096 --config.optim.iterations 100000
O_hybrid:
	python run.py --config examples/pp/lapnet/configs/hph/XO.py:Sc,1,0.21_187,hybrid --config.batch_size 4096 --config.optim.iterations 100000
O_l2:
	python run.py --config examples/pp/lapnet/configs/ph/XO.py:Sc,1,0.21_187,l2 --config.batch_size 4096 --config.optim.iterations 100000
